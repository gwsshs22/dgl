#include "executor_control_actor.h"

#include <algorithm>

#include <dgl/array.h>

#include "task_executors.h"

namespace dgl {
namespace inference {

namespace {
void fetch_result_fn(caf::blocking_actor* self,
                     const caf::actor& scheduler,
                     const caf::actor& executor,
                     const caf::actor& mpi_actor,
                     int batch_id,
                     int node_rank,
                     int local_rank) {
  if (node_rank == 0) {
    auto rh = self->request(executor, caf::infinite, caf::direct_fetch_result_atom_v, batch_id, local_rank);
    auto result = receive_result<std::vector<NDArray>>(rh);
    assert(result.size() == 1);
    self->send(scheduler, caf::finished_atom_v, batch_id, result[0]);
    return;
  }

  self->send(executor, caf::fetch_result_atom_v, batch_id, local_rank);
  auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, node_rank, CreateMpiTag(batch_id, TaskType::kFetchResult));
  auto result = receive_result<NDArray>(rh);
  self->send(scheduler, caf::finished_atom_v, batch_id, result);
}

void copy_fn_(caf::blocking_actor* self,
              const caf::actor& parent,
              const NDArray& dst,
              const NDArray& src,
              int feature_size,
              int pos) {

  if (src->shape[0] == 0) {
    self->send(parent, caf::done_atom_v);
    return;
  }

  size_t bytes_per_item = ((dst->dtype.bits * dst->dtype.lanes + 7) / 8) * feature_size;
  size_t total_copy_bytes = bytes_per_item * src->shape[0];
  assert(total_copy_bytes == src.GetSize());

  size_t offset = pos * bytes_per_item;
  void* dst_addr = (((u_int8_t*)dst->data) + offset);

  memcpy(dst_addr, src->data, total_copy_bytes);
  self->send(parent, caf::done_atom_v);
}

void broadcast_fetch_result_fn(caf::blocking_actor* self,
                               const caf::actor& scheduler,
                               const std::vector<caf::actor>& executors,
                               const caf::actor& mpi_actor,
                               const int num_nodes,
                               const int num_devices_per_node,
                               int batch_id) {

  auto rhs = std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>>();
  for (int i = 0; i < num_nodes; i++) {
    if (i == 0) {
      rhs.push_back(self->request(executors[i], caf::infinite, caf::direct_fetch_result_atom_v, batch_id, -1));
    } else {
      self->send(executors[i], caf::fetch_result_atom_v, batch_id, -1);
      for (int j = 0; j < num_devices_per_node; j++) {
        rhs.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, i, CreateMpiTag(batch_id, TaskType::kFetchResult, i, j)));
      }
    }
  }

  auto local_results = receive_result<std::vector<NDArray>>(rhs[0]);

  int rhs_idx = 1;
  for (int i = 1; i < num_nodes; i++) {
    for (int j = 0; j < num_devices_per_node; j++) {
      local_results.push_back(receive_result<NDArray>(rhs[rhs_idx++]));
    }
  }

  int batch_size = 0;
  int feature_size = -1;
  bool found_non_zero = false;
  uint8_t code = -1;
  uint8_t bits = -1;
  uint16_t lanes = -1;
  auto pos_list = std::vector<int64_t>();
  for (const auto& local_result : local_results) {
    assert(local_result->ndim == 2);

    pos_list.push_back(batch_size);
    int64_t local_batch_size = *local_result->shape;
    batch_size += local_batch_size;

    if (local_batch_size == 0) {
      continue;
    }

    if (!found_non_zero) {
      found_non_zero = true;
      feature_size = *(local_result->shape + 1);
      code = local_result->dtype.code;
      bits = local_result->dtype.bits;
      lanes = local_result->dtype.lanes;
    } else {
      assert(feature_size == *(local_result->shape + 1));
      assert(code == local_result->dtype.code);
      assert(bits == local_result->dtype.bits);
      assert(lanes == local_result->dtype.lanes);
    }
  }

  assert(feature_size > 0);
  auto result = NDArray::Empty(std::vector<int64_t>({ batch_size, feature_size }), DLDataType { code, bits, lanes }, DLContext { kDLCPU, 0 });
  auto self_ref = caf::actor_cast<caf::actor>(self->address());
  for (int i = 0; i < num_nodes * num_devices_per_node; i++) {
    self->spawn(copy_fn_, self_ref, result, local_results[i], feature_size, pos_list[i]); 
  }

  int loop_idx = 0;
  self->receive_for(loop_idx, num_nodes * num_devices_per_node) ([&](caf::done_atom) {});
  self->send(scheduler, caf::finished_atom_v, batch_id, result);
}
}

executor_control_actor::executor_control_actor(caf::actor_config& config,
                                               caf::strong_actor_ptr mpi_actor_ptr,
                                               int num_devices_per_node)
    : event_based_actor(config), num_devices_per_node_(num_devices_per_node) {
  mpi_actor_ = caf::actor_cast<caf::actor>(mpi_actor_ptr);
}

caf::behavior executor_control_actor::make_behavior() {
  return initializing();
}

caf::behavior executor_control_actor::initializing() {
  return {
    [&](caf::set_atom) { // From scheduler
      scheduler_connected_ = true;
      scheduler_actor_ = caf::actor_cast<caf::actor>(current_sender());
      TryRunning();
    },
    [&](caf::initialized_atom, const caf::strong_actor_ptr& executor_ptr, int executor_rank, int num_nodes) { // From executors
      assert(num_nodes >= 1);
      assert(executor_rank >= 0);
      assert(executor_rank < num_nodes);
      for (auto& p : pending_executors_) {
        assert(p.second != executor_rank);
      }

      if (num_nodes_ < 0) {
        num_nodes_ = num_nodes;
      } else {
        assert(num_nodes == num_nodes_);
      }

      pending_executors_.push_back(std::make_pair(executor_ptr, executor_rank));
      TryRunning();
    }
  };
}

void executor_control_actor::TryRunning() {
  if ((pending_executors_.size() == num_nodes_) && scheduler_connected_) {
    std::sort(pending_executors_.begin(), pending_executors_.end(),
      [](const auto& e1, const auto& e2) { return e1.second < e2.second; });

    for (auto const& pe : pending_executors_) {
      executors_.emplace_back(caf::actor_cast<caf::actor>(pe.first));
    }

    become(running());
    pending_executors_.clear();
    ReportToInitMon(*this, "exec_ctrl", 0, 1);
  }
}

caf::behavior executor_control_actor::running() {
  return {
    [&](caf::init_atom,
        int batch_id, 
        int node_rank,
        const NDArray& new_gnids,
        const NDArray& new_features,
        const NDArray& src_gnids,
        const NDArray& dst_gnids) {
      if (node_rank == 0) {
        send(executors_[node_rank], caf::init_atom_v, batch_id, new_gnids, new_features, src_gnids, dst_gnids);  
      } else {
        spawn(input_send_fn, mpi_actor_, node_rank, new_gnids, new_features, src_gnids, dst_gnids, CreateMpiTag(batch_id, TaskType::kInitialize));
        send(executors_[node_rank], caf::init_atom_v, batch_id);
      }

      done_task_counter_.emplace(std::make_pair(TaskType::kInitialize, batch_id), 1);
    },
    [&](caf::exec_atom, TaskType task_type, int batch_id, int node_rank, int local_rank) {
      send(executors_[node_rank], caf::exec_atom_v, task_type, batch_id, local_rank);
      done_task_counter_.emplace(std::make_pair(task_type, batch_id), 1);
    },
    [&](caf::fetch_result_atom, int batch_id, int node_rank, int local_rank) {
      spawn(fetch_result_fn, 
            scheduler_actor_,
            executors_[node_rank],
            mpi_actor_,
            batch_id,
            node_rank,
            local_rank);
    },
    [&](caf::broadcast_init_atom,
        int batch_id,
        const NDArray& new_gnids,
        const NDArray& new_features,
        const NDArray& src_gnids,
        const NDArray& dst_gnids) {
      send(executors_[0], caf::broadcast_init_atom_v, batch_id, new_gnids, new_features, src_gnids, dst_gnids);
      
      for (int i = 1; i < num_nodes_; i++) {
        send(executors_[i], caf::broadcast_init_atom_v, batch_id);
      }

      done_task_counter_.emplace(std::make_pair(TaskType::kInitialize, batch_id), num_nodes_);
    },
    [&](caf::broadcast_exec_atom, TaskType task_type, int batch_id) {
      for (int i = 0; i < num_nodes_; i++) {
        send(executors_[i], caf::exec_atom_v, task_type, batch_id, -1);
      }

      done_task_counter_.emplace(std::make_pair(task_type, batch_id), num_nodes_);
    },
    [&](caf::broadcast_fetch_result_atom, int batch_id) {
      spawn(broadcast_fetch_result_fn,
            scheduler_actor_,
            executors_,
            mpi_actor_,
            num_nodes_,
            num_devices_per_node_,
            batch_id);
    },
    [&](caf::done_atom, TaskType task_type, int batch_id, int node_rank) {
      auto p = std::make_pair(task_type, batch_id);
      auto it = done_task_counter_.find(p);
      ((*it).second)--;
      if (it->second > 0) {
        return;
      }

      done_task_counter_.erase(it);
      if (task_type == TaskType::kInitialize) {
        send(scheduler_actor_, caf::initialized_atom_v, batch_id);
      } else {
        send(scheduler_actor_, caf::done_atom_v, task_type, batch_id);
      }
        
    },
    
  };
}

}
}
