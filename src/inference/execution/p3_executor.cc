#include "executor_actor.h"

#include "./sampling/sampling_actor.h"
#include "../process/process_control_actor.h"

#include "mem_utils.h"
#include "array_utils.h"

namespace dgl {
namespace inference {

namespace {

void bsend_fn(caf::blocking_actor* self,
              const caf::actor& mpi_actor,
              const std::string name,
              int batch_id,
              int task_id) {
  auto tag = CreateMpiTag(batch_id, TaskType::kCompute, 0, 0, task_id);
  auto arr = LoadFromSharedMemory(batch_id, name);
  auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, arr, tag);
  receive_result<bool>(rh);
  self->receive([](caf::get_atom) { });
}

void brecv_fn(caf::blocking_actor* self,
              const caf::actor& mpi_actor,
              const caf::actor& object_storage_actor,
              const std::string name,
              int owner_node_rank,
              int batch_id,
              int task_id) {
  auto tag = CreateMpiTag(batch_id, TaskType::kCompute, 0, 0, task_id);
  auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, owner_node_rank, tag);
  auto arr = receive_result<NDArray>(rh);

  auto rh2 = self->request(object_storage_actor, caf::infinite, caf::put_atom_v, name, arr);
  receive_result<bool>(rh2);
  rh2 = self->request(object_storage_actor, caf::infinite, caf::move_to_shared_atom_v, name);
  receive_result<bool>(rh2);

  self->receive([](caf::get_atom) { });
}

void p3_push_comp_graph(caf::blocking_actor* self,
                        const caf::actor& mpi_actor,
                        const caf::actor& trace_actor,
                        const caf::actor& object_storage_actor,
                        int batch_id,
                        int node_rank,
                        int owner_node_rank) {
  if (node_rank == owner_node_rank) {
    auto start_time = std::chrono::steady_clock::now();

    std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;
    auto bsend_lambda = [&](const std::string& name, int task_id) {
      auto actor = self->spawn(bsend_fn, mpi_actor, name, batch_id, task_id);
      auto rh = self->request(actor, caf::infinite, caf::get_atom_v);
      return rh;
    };

    rh_list.push_back(bsend_lambda("input_gnids", 0));
    rh_list.push_back(bsend_lambda("b1_u", 1));
    rh_list.push_back(bsend_lambda("b1_v", 2));
    rh_list.push_back(bsend_lambda("num_src_nodes_list", 3));
    rh_list.push_back(bsend_lambda("num_dst_nodes_list", 4));
    for (int i = 0; i < rh_list.size(); i++) {
      receive_result<void>(rh_list[i]);
    }

    if (TRACE_ENABLED) {
      int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
      self->send(trace_actor, caf::put_atom_v, batch_id, "push_comp_graph", elapsed);
    }
  } else {
    std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;
    auto brecv_lambda = [&](const std::string& name, int task_id) {
      auto actor = self->spawn(brecv_fn, mpi_actor, object_storage_actor, name, owner_node_rank, batch_id, task_id);
      auto rh = self->request(actor, caf::infinite, caf::get_atom_v);
      return rh;
    };

    rh_list.push_back(brecv_lambda("input_gnids", 0));
    rh_list.push_back(brecv_lambda("b1_u", 1));
    rh_list.push_back(brecv_lambda("b1_v", 2));
    rh_list.push_back(brecv_lambda("num_src_nodes_list", 3));
    rh_list.push_back(brecv_lambda("num_dst_nodes_list", 4));

    for (int i = 0; i < rh_list.size(); i++) {
      receive_result<void>(rh_list[i]);
    }
  }

  self->receive([](caf::get_atom) { });
}

void p3_compute(caf::blocking_actor* self,
                const caf::actor& mpi_actor,
                const caf::actor& gnn_executor_group,
                int batch_id,
                int node_rank,
                int num_devices_per_node,
                int owner_node_rank,
                int owner_local_rank) {
  int owner_gpu_global_rank = num_devices_per_node * owner_node_rank + owner_local_rank;

  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;
  for (int local_rank = 0; local_rank < num_devices_per_node; local_rank++) {
    if (node_rank == owner_node_rank && local_rank == owner_local_rank) {
      auto rh = self->request(gnn_executor_group,
                              caf::infinite,
                              caf::exec_atom_v,
                              batch_id,
                              static_cast<int>(gnn_executor_request_type::kP3OwnerComputeRequest),
                              local_rank,
                              owner_gpu_global_rank);
      rh_list.push_back(rh);
    } else {
      auto rh = self->request(gnn_executor_group,
                              caf::infinite,
                              caf::exec_atom_v,
                              batch_id,
                              static_cast<int>(gnn_executor_request_type::kP3OtherComputeRequest),
                              local_rank,
                              owner_gpu_global_rank);
      rh_list.push_back(rh);
    }
  }

  for (auto& rh : rh_list) {
    receive_result<bool>(rh);
  }

  self->receive([](caf::get_atom) { });
}

void direct_fetch_result_fn(caf::blocking_actor* self,
                            int batch_id,
                            int local_rank) {
  
  auto src_arr = LoadFromSharedMemory(batch_id, "result");
  NDArray copied = NDArray::Empty(
      std::vector<int64_t>(src_arr->shape, src_arr->shape + src_arr->ndim),
      src_arr->dtype,
      DLContext{kDLCPU, 0});

  copied.CopyFrom(src_arr);
  self->receive([=](caf::get_atom) {
    return copied;
  });
};

void fetch_result_fn(caf::blocking_actor* self,
                     const caf::actor& mpi_actor,
                     int batch_id,
                     int local_rank) {
  auto result = LoadFromSharedMemory(batch_id, "result");
  auto rh2 = self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, 0, result, CreateMpiTag(batch_id, TaskType::kFetchResult));
  receive_result<bool>(rh2);

  self->receive([](caf::get_atom) { });
}

void write_traces_fn(caf::blocking_actor* self,
                     const caf::actor& trace_actor,
                     const std::string& result_dir,
                     int num_devices_per_node,
                     int node_rank,
                     const caf::actor& gnn_executor_group,
                     const std::vector<caf::actor>& samplers,
                     caf::response_promise rp) {
  auto rh = self->request(trace_actor, caf::infinite, caf::write_trace_atom_v);
  receive_result<bool>(rh);

  for (int i = 0; i < samplers.size(); i++) {
    auto rh = self->request(samplers[i], caf::infinite, caf::write_trace_atom_v);
    receive_result<bool>(rh);
  }

  for (int i = 0; i < num_devices_per_node; i++) {
    auto rh = self->request(gnn_executor_group,
                            caf::infinite,
                            caf::exec_atom_v,
                            -1,
                            static_cast<int>(gnn_executor_request_type::kWriteTraces),
                            i,
                            -1);
    receive_result<bool>(rh);
  }

  rp.deliver(true);  
}
}

p3_executor::p3_executor(caf::actor_config& config,
                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                         caf::strong_actor_ptr mpi_actor_ptr,
                         caf::strong_actor_ptr trace_actor_ptr,
                         int node_rank,
                         int num_nodes,
                         int num_backup_servers,
                         int num_devices_per_node,
                         int num_samplers_per_node,
                         std::string result_dir,
                         bool collect_stats)
    : executor_actor(config,
                     exec_ctl_actor_ptr,
                     mpi_actor_ptr,
                     trace_actor_ptr,
                     node_rank,
                     num_nodes,
                     num_backup_servers,
                     num_devices_per_node,
                     result_dir,
                     collect_stats,
                     num_samplers_per_node) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  for (int i = 0; i < num_samplers_per_node; i++) {
    samplers_.emplace_back(spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, i));
    sampler_running_.push_back(false);
  }
}

FeatureSplitMethod p3_executor::GetFeatureSplit(int batch_size, int feature_size) {
  return GetP3FeatureSplit(num_nodes_, batch_size, feature_size);
}

void p3_executor::Sampling(int batch_id, int, int) {
  int idle_sampler_rank = -1;
  for (int i = 0; i < sampler_running_.size(); i++) {
    if (!sampler_running_[i]) {
      idle_sampler_rank = i;
      break;
    }
  }

  CHECK(idle_sampler_rank != -1);
  sampler_running_[idle_sampler_rank] = true;
  batch_id_to_sampler_rank_[batch_id] = idle_sampler_rank;
  auto sampling_task = spawn(sampling_fn, samplers_[idle_sampler_rank], batch_id);

  RequestAndReportTaskDone(sampling_task, TaskType::kSampling, batch_id, [&, idle_sampler_rank]() {
    sampler_running_[idle_sampler_rank] = false;
  });
}

void p3_executor::PushComputationGraph(int batch_id, int, int owner_node_rank, int) {
  auto object_storage_actor = object_storages_[batch_id];
  auto push_comp_graph_task = spawn(p3_push_comp_graph, mpi_actor_, trace_actor_, object_storage_actor, batch_id, node_rank_, owner_node_rank);
  RequestAndReportTaskDone(push_comp_graph_task, TaskType::kPushComputationGraph, batch_id);
}

void p3_executor::Compute(int batch_id, int, int owner_node_rank, int owner_local_rank) {
  if (owner_node_rank == node_rank_) {
    batch_id_to_gpu_local_rank_[batch_id] = owner_local_rank;
  }

  auto compute_task = spawn(p3_compute, mpi_actor_, gnn_executor_group_, batch_id, node_rank_, num_devices_per_node_, owner_node_rank, owner_local_rank);
  RequestAndReportTaskDone(compute_task, TaskType::kCompute, batch_id);
}

void p3_executor::DirectFetchResult(int batch_id,
                                    int local_rank,
                                    caf::response_promise rp) {
  auto task = spawn(direct_fetch_result_fn, batch_id, local_rank);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=](NDArray result) mutable {
      rp.deliver(std::vector<NDArray>({result}));
    },
    [=](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << "DirectFetchResult (batch_id=" << batch_id << "): " << caf::to_string(err) << std::endl;
    });
}

void p3_executor::FetchResult(int batch_id, int local_rank) {
  auto task = spawn(fetch_result_fn, mpi_actor_, batch_id, local_rank);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() {  },
    [=](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << "FetchResult (batch_id=" << batch_id << "): " << caf::to_string(err) << std::endl;
    });
}

void p3_executor::Cleanup(int batch_id, int, int) {
  auto it = batch_id_to_gpu_local_rank_.find(batch_id);
  if (it != batch_id_to_gpu_local_rank_.end()) {
    auto local_rank = it->second;
    send(gnn_executor_group_,
        caf::exec_atom_v,
        batch_id,
        static_cast<int>(gnn_executor_request_type::kCleanupRequestType),
        local_rank,
        /* param0 */ -1);

    batch_id_to_gpu_local_rank_.erase(it);
  }

  auto sampler_it = batch_id_to_sampler_rank_.find(batch_id);
  if (sampler_it != batch_id_to_sampler_rank_.end()) {
    send(samplers_[sampler_it->second], caf::cleanup_atom_v, batch_id);
    batch_id_to_sampler_rank_.erase(sampler_it);
  }

  object_storages_.erase(batch_id);
  ReportTaskDone(TaskType::kCleanup, batch_id);
}

void p3_executor::WriteExecutorTraces(caf::response_promise rp) {
  spawn(write_traces_fn, trace_actor_, result_dir_, num_devices_per_node_, node_rank_, gnn_executor_group_, samplers_, rp);
}

}
}
