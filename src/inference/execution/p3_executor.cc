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
              const std::string name,
              int owner_node_rank,
              int batch_id,
              int task_id) {
  auto tag = CreateMpiTag(batch_id, TaskType::kCompute, 0, 0, task_id);
  auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, owner_node_rank, tag);
  auto arr = receive_result<NDArray>(rh);
  auto metadata_name = GetArrayMetadataName(batch_id, name);
  assert(!runtime::SharedMemory::Exist(metadata_name));
  auto meta_holder = CreateMetadataSharedMem(metadata_name, arr);

  assert(arr->ctx.device_type == kDLCPU);
  auto arr_holder = CopyToSharedMem(batch_id, name, arr);
  self->receive([=](caf::get_atom) {
    return std::make_pair(arr_holder, meta_holder);
  });
}

void p3_compute(caf::blocking_actor* self,
                const caf::actor& mpi_actor,
                const caf::actor& gnn_executor_group,
                int batch_id,
                int num_nodes,
                int node_rank,
                int num_devices_per_node,
                int owner_node_rank,
                int owner_local_rank) {
  // For receive case. We should hold the shared memory until GNN computation finishes.
  NDArray input_gnids;
  std::shared_ptr<runtime::SharedMemory> input_gnids_meta;
  NDArray b1_u;
  std::shared_ptr<runtime::SharedMemory> b1_u_meta;
  NDArray b1_v;
  std::shared_ptr<runtime::SharedMemory> b1_v_meta;
  NDArray num_src_nodes_list;
  std::shared_ptr<runtime::SharedMemory> num_src_nodes_list_meta;
  NDArray num_dst_nodes_list;
  std::shared_ptr<runtime::SharedMemory> num_dst_nodes_list_meta;

  if (node_rank == owner_node_rank) {
    TraceMe push_comp_graph(batch_id, "push_comp_graph");

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
  } else {
    std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;
    auto brecv_lambda = [&](const std::string& name, int task_id) {
      auto actor = self->spawn(brecv_fn, mpi_actor, name, owner_node_rank, batch_id, task_id);
      auto rh = self->request(actor, caf::infinite, caf::get_atom_v);
      return rh;
    };

    rh_list.push_back(brecv_lambda("input_gnids", 0));
    rh_list.push_back(brecv_lambda("b1_u", 1));
    rh_list.push_back(brecv_lambda("b1_v", 2));
    rh_list.push_back(brecv_lambda("num_src_nodes_list", 3));
    rh_list.push_back(brecv_lambda("num_dst_nodes_list", 4));

    auto input_gnids_result = receive_result<NDArrayWithSharedMeta>(rh_list[0]);
    input_gnids = input_gnids_result.first;
    input_gnids_meta = input_gnids_result.second;

    auto b1_u_result = receive_result<NDArrayWithSharedMeta>(rh_list[1]);
    b1_u = b1_u_result.first;
    b1_u_meta = b1_u_result.second;

    auto b1_v_result = receive_result<NDArrayWithSharedMeta>(rh_list[2]);
    b1_v = b1_v_result.first;
    b1_v_meta = b1_v_result.second;

    auto num_src_nodes_list_result = receive_result<NDArrayWithSharedMeta>(rh_list[3]);
    num_src_nodes_list = num_src_nodes_list_result.first;
    num_src_nodes_list_meta = num_src_nodes_list_result.second;

    auto num_dst_nodes_list_result = receive_result<NDArrayWithSharedMeta>(rh_list[4]);
    num_dst_nodes_list = num_dst_nodes_list_result.first;
    num_dst_nodes_list_meta = num_dst_nodes_list_result.second;
  }

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
                     const std::string& result_dir,
                     int num_devices_per_node,
                     int node_rank,
                     const caf::actor& gnn_executor_group,
                     const std::vector<caf::actor>& samplers,
                     caf::response_promise rp) {
  WriteTraces(result_dir, node_rank);

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
                         int node_rank,
                         int num_nodes,
                         int num_backup_servers,
                         int num_devices_per_node,
                         std::string result_dir,
                         bool collect_stats)
    : executor_actor(config,
                     exec_ctl_actor_ptr,
                     mpi_actor_ptr,
                     node_rank,
                     num_nodes,
                     num_backup_servers,
                     num_devices_per_node,
                     result_dir,
                     collect_stats,
                     num_devices_per_node) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  for (int i = 0; i < num_devices_per_node; i++) {
    samplers_.emplace_back(spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, i));
  }
}

FeatureSplitMethod p3_executor::GetFeatureSplit(int batch_size, int feature_size) {
  return GetP3FeatureSplit(num_nodes_, batch_size, feature_size);
}

void p3_executor::Sampling(int batch_id, int local_rank) {
  assigned_local_rank_[batch_id] = local_rank;
  auto sampling_task = spawn(sampling_fn, samplers_[local_rank], batch_id);
  RequestAndReportTaskDone(sampling_task, TaskType::kSampling, batch_id);
}

void p3_executor::PrepareInput(int batch_id, int local_rank) {
  // Do nothing.
  ReportTaskDone(TaskType::kPrepareInput, batch_id);
}

void p3_executor::Compute(int batch_id, int, int owner_node_rank, int owner_local_rank) {
  auto compute_task = spawn(p3_compute, mpi_actor_, gnn_executor_group_, batch_id, num_nodes_, node_rank_, num_devices_per_node_, owner_node_rank, owner_local_rank);
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

void p3_executor::Cleanup(int batch_id, int) {
  auto it = assigned_local_rank_.find(batch_id);
  if (it != assigned_local_rank_.end()) {
    auto local_rank = it->second;
    send(samplers_[local_rank], caf::cleanup_atom_v, batch_id);

    send(gnn_executor_group_,
        caf::exec_atom_v,
        batch_id,
        static_cast<int>(gnn_executor_request_type::kCleanupRequestType),
        local_rank,
        /* param0 */ -1);

    assigned_local_rank_.erase(it);
  }

  object_storages_.erase(batch_id);
  ReportTaskDone(TaskType::kCleanup, batch_id);
}

void p3_executor::WriteExecutorTraces(caf::response_promise rp) {
  spawn(write_traces_fn, result_dir_, num_devices_per_node_, node_rank_, gnn_executor_group_, samplers_, rp);
}

}
}
