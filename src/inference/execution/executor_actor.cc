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
                               int num_backup_servers,
                               int num_devices_per_node,
                               std::string result_dir,
                               bool collect_stats,
                               int required_init_count)
    : event_based_actor(config),
      node_rank_(node_rank),
      num_nodes_(num_nodes),
      num_backup_servers_(num_backup_servers),
      num_devices_per_node_(num_devices_per_node),
      result_dir_(result_dir),
      collect_stats_(collect_stats),
      required_init_count_(required_init_count + 1 + (1 + num_backup_servers)) { // Include the common graph_server_actor and gnn_executor_group
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
  mpi_actor_ = caf::actor_cast<caf::actor>(mpi_actor_ptr);
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  gnn_executor_group_ = spawn<caf::linked + caf::monitored>(
        gnn_executor_group, self_ptr, num_devices_per_node);
  for (int i = 0; i < (1 + num_backup_servers); i++) {
    auto graph_server = spawn<graph_server_actor, caf::linked + caf::monitored>(self_ptr, i);
  }
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

caf::behavior executor_actor::make_running_behavior() {
  return {
    [&](caf::init_atom, int batch_id, const NDArray& new_gnids, const NDArray& new_features, const NDArray& src_gnids, const NDArray& dst_gnids) {
      InputRecvFromSameMachine(batch_id, new_gnids, new_features, src_gnids, dst_gnids);
    },
    [&](caf::init_atom, int batch_id) {
      InputRecv(batch_id);
    },
    [&](caf::broadcast_init_atom, BroadcastInitType init_type, int batch_id, const NDArray& new_gnids, const NDArray& new_features, const NDArray& src_gnids, const NDArray& dst_gnids) {
      BroadcastInputSend(init_type, batch_id, new_gnids, new_features, src_gnids, dst_gnids);
    },
    [&](caf::broadcast_init_atom, BroadcastInitType init_type, int batch_id) {
      BroadcastInputRecv(init_type, batch_id);
    },
    // batch task execution
    [&](caf::exec_atom, TaskType task_type, int batch_id, int local_rank, int param0, int param1) {
      switch (task_type) {
        case TaskType::kSampling:
          Sampling(batch_id, local_rank, param0);
          break;
        case TaskType::kPushComputationGraph:
          PushComputationGraph(batch_id, local_rank, param0, param1);
          break;
        case TaskType::kCompute:
          Compute(batch_id, local_rank, param0, param1);
          break;
        case TaskType::kCleanup:
          Cleanup(batch_id, local_rank, param0);
          break;
      }
    },
    [&](caf::direct_fetch_result_atom, int batch_id, int local_rank) {
      auto rp = make_response_promise<std::vector<NDArray>>();
      DirectFetchResult(batch_id, local_rank, rp);
      return rp;
    },
    [&](caf::fetch_result_atom, int batch_id, int local_rank) {
      FetchResult(batch_id, local_rank);
    },
    [&](caf::write_trace_atom) {
      auto rp = make_response_promise<bool>();
      WriteExecutorTraces(rp);
      return rp;
    }
  };
}

void executor_actor::InputRecvFromSameMachine(int batch_id,
                                              const NDArray& new_gnids,
                                              const NDArray& new_features,
                                              const NDArray& src_gnids,
                                              const NDArray& dst_gnids) {
  auto obj_storage_actor = spawn(object_storage_actor, batch_id);
  object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
  auto shared_mem_copier = spawn(move_input_to_shared_mem_fn, obj_storage_actor, batch_id, new_gnids, new_features, src_gnids, dst_gnids);
  RequestAndReportTaskDone(shared_mem_copier, TaskType::kInitialize, batch_id);
}

void executor_actor::InputRecv(int batch_id) {
  auto obj_storage_actor = spawn(object_storage_actor, batch_id);
  object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
  auto receiver = spawn(input_recv_fn, mpi_actor_, obj_storage_actor, batch_id, CreateMpiTag(batch_id, TaskType::kInitialize));
  RequestAndReportTaskDone(receiver, TaskType::kInitialize, batch_id);
}

void executor_actor::BroadcastInputSend(BroadcastInitType init_type,
                                        int batch_id,
                                        const NDArray& new_gnids,
                                        const NDArray& new_features,
                                        const NDArray& src_gnids,
                                        const NDArray& dst_gnids) {
  auto obj_storage_actor = spawn(object_storage_actor, batch_id);
  object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
  caf::actor broadcaster;
  if (init_type == BroadcastInitType::kAll) {
    broadcaster = spawn(input_bsend_all_fn,
                        mpi_actor_,
                        obj_storage_actor,
                        batch_id,
                        new_gnids,
                        new_features,
                        src_gnids,
                        dst_gnids,
                        CreateMpiTag(batch_id, TaskType::kInitialize));
  } else {
    broadcaster = spawn(input_bsend_scatter_fn,
                        mpi_actor_,
                        obj_storage_actor,
                        num_nodes_,
                        init_type,
                        batch_id,
                        new_gnids,
                        new_features,
                        src_gnids,
                        dst_gnids,
                        GetFeatureSplit(new_features->shape[0], new_features->shape[1]),
                        CreateMpiTag(batch_id, TaskType::kInitialize));
  }

  RequestAndReportTaskDone(broadcaster, TaskType::kInitialize, batch_id);  
}

void executor_actor::BroadcastInputRecv(BroadcastInitType init_type, int batch_id) {
  auto obj_storage_actor = spawn(object_storage_actor, batch_id);
  object_storages_.emplace(std::make_pair(batch_id, caf::actor_cast<caf::actor>(obj_storage_actor)));
  caf::actor receiver;
  if (init_type == BroadcastInitType::kAll){
    receiver = spawn(input_brecv_all_fn, mpi_actor_, obj_storage_actor, batch_id, CreateMpiTag(batch_id, TaskType::kInitialize));
  } else {
    receiver = spawn(input_brecv_scatter_fn, mpi_actor_, obj_storage_actor, node_rank_, init_type, batch_id, CreateMpiTag(batch_id, TaskType::kInitialize));
  }

  RequestAndReportTaskDone(receiver, TaskType::kInitialize, batch_id);
}

caf::actor spawn_executor_actor(caf::actor_system& system,
                                ParallelizationType parallelization_type,
                                const caf::strong_actor_ptr& exec_ctl_actor_ptr,
                                const caf::strong_actor_ptr& mpi_actor_ptr,
                                int node_rank,
                                int num_nodes,
                                int num_backup_servers,
                                int num_devices_per_node,
                                int num_samplers_per_node,
                                std::string result_dir,
                                bool collect_stats,
                                bool using_precomputed_aggs) {
  if (parallelization_type == ParallelizationType::kData) {
    return system.spawn<data_parallel_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_backup_servers, num_devices_per_node, num_samplers_per_node, result_dir, collect_stats);
  } else if (parallelization_type == ParallelizationType::kP3) {
    return system.spawn<p3_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_backup_servers, num_devices_per_node, num_samplers_per_node, result_dir, collect_stats);
  } else {
    return system.spawn<vertex_cut_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_backup_servers, num_devices_per_node, num_samplers_per_node, result_dir, collect_stats, using_precomputed_aggs);
  }
}

}
}
