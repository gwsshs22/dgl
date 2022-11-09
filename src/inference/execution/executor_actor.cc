#include "executor_actor.h"

#include "./gnn/gnn_executor.h"
#include "./gnn/graph_server_actor.h"
#include "task_executors.h"

namespace dgl {
namespace inference {


executor_actor::executor_actor(caf::actor_config& config,
                               caf::strong_actor_ptr exec_ctl_actor_ptr,
                               caf::strong_actor_ptr mpi_actor_ptr,
                               int node_rank,
                               int num_nodes,
                               int num_devices_per_node,
                               int required_init_count)
    : event_based_actor(config),
      node_rank_(node_rank),
      num_nodes_(num_nodes),
      required_init_count_(required_init_count + 2) { // Include the common graph_server_actor and gnn_executor_group
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
  mpi_actor_ = caf::actor_cast<caf::actor>(mpi_actor_ptr);
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  gnn_executor_group_ = spawn<caf::linked + caf::monitored>(
        gnn_executor_group, self_ptr, num_devices_per_node);
  graph_server_actor_ = spawn<graph_server_actor, caf::linked + caf::monitored>(self_ptr);
}

caf::behavior executor_actor::make_behavior() {
  return make_initializing_behavior();
}

// Initializing
caf::behavior executor_actor::make_initializing_behavior() {
  return {
    [&](caf::initialized_atom, const std::string component_name, int) {
      num_initialized_components_ += 1;
      if (num_initialized_components_ == required_init_count_) {
        // All components are initialized.
        send(exec_ctl_actor_, caf::initialized_atom_v, caf::actor_cast<caf::strong_actor_ptr>(this), node_rank_, num_nodes_);
        become(make_running_behavior());
      }
    }
  };
}

// Running
void executor_actor::ReportTaskDone(TaskType task_type, int batch_id) {
  send(exec_ctl_actor_, caf::done_atom_v, task_type, batch_id, node_rank_);
}

template <typename T, typename F>
void executor_actor::RequestAndReportTaskDone(caf::actor& task_executor,
                                              TaskType task_type,
                                              int batch_id,
                                              F&& callback) {
  request(task_executor, caf::infinite, caf::get_atom_v).then(
    [=](const T& ret) {
      callback(ret);
      ReportTaskDone(task_type, batch_id);
    },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

inline void executor_actor::RequestAndReportTaskDone(caf::actor& task_executor,
                                              TaskType task_type,
                                              int batch_id) {
  request(task_executor, caf::infinite, caf::get_atom_v).then(
    [=]{ ReportTaskDone(task_type, batch_id); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

caf::behavior executor_actor::make_running_behavior() {
  return {
    [&](caf::init_atom, int batch_id, const NDArray& new_gnids, const NDArray& src_gnids, const NDArray& dst_gnids) {
      auto obj_storage_actor = spawn<caf::monitored>(object_storage_actor, batch_id);
      object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
      auto shared_mem_copier = spawn(move_input_to_shared_mem_fn, obj_storage_actor, new_gnids, src_gnids, dst_gnids);
      RequestAndReportTaskDone(shared_mem_copier, TaskType::kInitialize, batch_id);
    },
    [&](caf::init_atom, int batch_id) {
      auto obj_storage_actor = spawn<caf::monitored>(object_storage_actor, batch_id);
      object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
      auto receiver = spawn(input_recv_fn, mpi_actor_, CreateMpiTag(batch_id, TaskType::kInitialize));
      request(receiver, caf::infinite, caf::get_atom_v).then(
        [=](const std::vector<NDArray>& ret) {
          auto shared_mem_copier = spawn(move_input_to_shared_mem_fn, obj_storage_actor, ret[0], ret[1], ret[2]);
          RequestAndReportTaskDone(shared_mem_copier, TaskType::kInitialize, batch_id);
        },
        [&](caf::error& err) {
          // TODO: error handling
          caf::aout(this) << caf::to_string(err) << std::endl;
        });
    },
    [&](caf::broadcast_init_atom, int batch_id, const NDArray& new_gnids, const NDArray& src_gnids, const NDArray& dst_gnids) {
      auto obj_storage_actor = spawn<caf::monitored>(object_storage_actor, batch_id);
      object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
      auto broadcaster = spawn(input_bsend_fn, mpi_actor_, new_gnids, src_gnids, dst_gnids, CreateMpiTag(batch_id, TaskType::kInitialize));
      request(broadcaster, caf::infinite, caf::get_atom_v).then(
        [=]() {
          auto shared_mem_copier = spawn(move_input_to_shared_mem_fn, obj_storage_actor, new_gnids, src_gnids, dst_gnids);
          RequestAndReportTaskDone(shared_mem_copier, TaskType::kInitialize, batch_id);
        },
        [&](caf::error& err) {
          // TODO: error handling
          caf::aout(this) << caf::to_string(err) << std::endl;
        });
    },
    [&](caf::broadcast_init_atom, int batch_id) {
      auto obj_storage_actor = spawn<caf::monitored>(object_storage_actor, batch_id);
      object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
      auto receiver = spawn(input_brecv_fn, mpi_actor_, CreateMpiTag(batch_id, TaskType::kInitialize));
      request(receiver, caf::infinite, caf::get_atom_v).then(
        [=](const std::vector<NDArray>& ret) {
          auto shared_mem_copier = spawn(move_input_to_shared_mem_fn, obj_storage_actor, ret[0], ret[1], ret[2]);
          RequestAndReportTaskDone(shared_mem_copier, TaskType::kInitialize, batch_id);
        },
        [&](caf::error& err) {
          // TODO: error handling
          caf::aout(this) << caf::to_string(err) << std::endl;
        });
    },
    // batch task execution
    [&](caf::exec_atom, TaskType task_type, int batch_id, int local_rank) {
      switch (task_type) {
        case TaskType::kSampling:
          Sampling(batch_id, local_rank);
          break;
        case TaskType::kPrepareInput:
          PrepareInput(batch_id, local_rank);
          break;
        case TaskType::kCompute:
          Compute(batch_id, local_rank);
          break;
        case TaskType::kPrepareAggregations:
          PrepareAggregations(batch_id, local_rank);
          break;
        case TaskType::kRecomputeAggregations:
          RecomputeAggregations(batch_id, local_rank);
          break;
        case TaskType::kComputeRemaining:
          ComputeRemaining(batch_id, local_rank);
          break;
      }
    }
  };
}

caf::actor spawn_executor_actor(caf::actor_system& system,
                                ParallelizationType parallelization_type,
                                const caf::strong_actor_ptr& exec_ctl_actor_ptr,
                                const caf::strong_actor_ptr& mpi_actor_ptr,
                                int node_rank,
                                int num_nodes,
                                int num_devices_per_node,
                                bool using_precomputed_aggs) {
  if (parallelization_type == ParallelizationType::kData) {
    return system.spawn<data_parallel_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_devices_per_node);
  } else if (parallelization_type == ParallelizationType::kP3) {
    return system.spawn<p3_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_devices_per_node);
  } else {
    return system.spawn<vertex_cut_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_devices_per_node, using_precomputed_aggs);
  }
}

}
}
