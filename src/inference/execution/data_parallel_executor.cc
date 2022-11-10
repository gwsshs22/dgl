#include "executor_actor.h"

#include "./sampling/sampling_actor.h"
#include "../process/process_control_actor.h"

#include "mem_utils.h"

namespace dgl {
namespace inference {

namespace {

void direct_fetch_result_fn(caf::blocking_actor* self,
                            int batch_id,
                            int local_rank,
                            caf::response_promise rp) {
  auto result = LoadFromSharedMemory(batch_id, "result");
  rp.deliver(result);

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

data_parallel_executor::data_parallel_executor(caf::actor_config& config,
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

void data_parallel_executor::Sampling(int batch_id, int local_rank) {
  auto sampling_task = spawn(sampling_fn, samplers_[local_rank], batch_id);
  RequestAndReportTaskDone(sampling_task, TaskType::kSampling, batch_id);
}

void data_parallel_executor::PrepareInput(int batch_id, int local_rank) {
  // Do nothing.
  ReportTaskDone(TaskType::kPrepareInput, batch_id);
}

void data_parallel_executor::Compute(int batch_id, int local_rank) {
  auto compute_task = spawn(gnn_execute_fn, gnn_executor_group_, batch_id, gnn_executor_request_type::kComputeRequestType, local_rank);
  RequestAndReportTaskDone(compute_task, TaskType::kCompute, batch_id);
}

void data_parallel_executor::DirectFetchResult(int batch_id,
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

void data_parallel_executor::FetchResult(int batch_id, int local_rank) {
  auto task = spawn(fetch_result_fn, mpi_actor_, batch_id, local_rank);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { this->Cleanup(batch_id, local_rank); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void data_parallel_executor::Cleanup(int batch_id, int local_rank) {
  send(samplers_[local_rank], caf::cleanup_atom_v, batch_id);
  send(gnn_executor_group_,
       caf::exec_atom_v,
       batch_id,
       static_cast<int>(gnn_executor_request_type::kCleanupRequestType),
       local_rank);

  object_storages_.erase(batch_id);
}

}
}
