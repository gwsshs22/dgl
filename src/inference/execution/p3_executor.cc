#include "executor_actor.h"

#include "./sampling/sampling_actor.h"
#include "../process/process_control_actor.h"

#include "mem_utils.h"

namespace dgl {
namespace inference {

namespace {

void p3_compute(caf::blocking_actor* self,
                const caf::actor& mpi_actor,
                const caf::actor& gnn_executor_group,
                int batch_id,
                int num_nodes,
                int node_rank,
                int num_devices_per_node,
                int owner_node_rank,
                int owner_local_rank) {
  uint32_t tag = CreateMpiTag(batch_id, TaskType::kCompute);

  // For receive case. We should hold the shared memory until GNN computation finishes.
  NDArray new_features;
  std::shared_ptr<runtime::SharedMemory> new_features_meta;
  NDArray input_gnids;
  std::shared_ptr<runtime::SharedMemory> input_gnids_meta;
  NDArray b1_u;
  std::shared_ptr<runtime::SharedMemory> b1_u_meta;
  NDArray b1_v;
  std::shared_ptr<runtime::SharedMemory> b1_v_meta;

  if (node_rank == owner_node_rank) {
    auto bsend_fn = [&](const NDArray& arr) {
      auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, arr, tag);
      receive_result<bool>(rh);
    };

    new_features = LoadFromSharedMemory(batch_id, "new_features");
    input_gnids = LoadFromSharedMemory(batch_id, "input_gnids");
    b1_u = LoadFromSharedMemory(batch_id, "b1_u");
    b1_v = LoadFromSharedMemory(batch_id, "b1_v");

    bsend_fn(new_features);
    bsend_fn(input_gnids);
    bsend_fn(b1_u);
    bsend_fn(b1_v);
  } else {
    auto brecv_fn = [&](const std::string& name, NDArray& arr_holder, std::shared_ptr<runtime::SharedMemory>& meta_holder) {
      auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, owner_node_rank, tag);
      auto arr = receive_result<NDArray>(rh);
      auto metadata_name = GetArrayMetadataName(batch_id, name);
      assert(!runtime::SharedMemory::Exist(metadata_name));
      meta_holder = CreateMetadataSharedMem(metadata_name, arr);

      assert(arr->ctx.device_type == kDLCPU);
      arr_holder = CopyToSharedMem(batch_id, name, arr);
    };

    brecv_fn("new_features", new_features, new_features_meta);
    brecv_fn("input_gnids", input_gnids, input_gnids_meta);
    brecv_fn("b1_u", b1_u, b1_u_meta);
    brecv_fn("b1_v", b1_v, b1_v_meta);
  }

  int owner_gpu_global_rank = num_nodes * owner_node_rank + owner_local_rank;

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
                            int local_rank,
                            caf::response_promise rp) {
  rp.deliver(std::vector<NDArray>({ LoadFromSharedMemory(batch_id, "result") }));

  self->receive([](caf::get_atom) { });
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

}

p3_executor::p3_executor(caf::actor_config& config,
                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                         caf::strong_actor_ptr mpi_actor_ptr,
                         int node_rank,
                         int num_nodes,
                         int num_devices_per_node)
    : executor_actor(config,
                     exec_ctl_actor_ptr,
                     mpi_actor_ptr,
                     node_rank,
                     num_nodes,
                     num_devices_per_node,
                     num_devices_per_node) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  for (int i = 0; i < num_devices_per_node; i++) {
    samplers_.emplace_back(spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, i));
  }
}

void p3_executor::Sampling(int batch_id, int local_rank) {
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
  auto task = spawn(direct_fetch_result_fn, batch_id, local_rank, rp);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { this->Cleanup(batch_id, local_rank); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void p3_executor::FetchResult(int batch_id, int local_rank) {
  auto task = spawn(fetch_result_fn, mpi_actor_, batch_id, local_rank);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { this->Cleanup(batch_id, local_rank); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void p3_executor::Cleanup(int batch_id, int local_rank) {
  send(samplers_[local_rank], caf::cleanup_atom_v, batch_id);
  send(gnn_executor_group_,
       caf::exec_atom_v,
       batch_id,
       static_cast<int>(gnn_executor_request_type::kCleanupRequestType),
       local_rank,
       /* param0 */ -1);

  object_storages_.erase(batch_id);
}

}
}
