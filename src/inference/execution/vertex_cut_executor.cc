#include "executor_actor.h"

#include <dgl/array.h>

#include "./sampling/sampling_actor.h"
#include "../process/process_control_actor.h"

#include "mem_utils.h"

namespace dgl {
namespace inference {

namespace {

void direct_fetch_result_fn(caf::blocking_actor* self,
                            int num_devices_per_node,
                            int batch_id,
                            caf::response_promise rp) {

  auto results = std::vector<NDArray>();
  for (int local_rank = 0; local_rank < num_devices_per_node; local_rank++) {
    results.push_back(LoadFromSharedMemory(batch_id, "g" + std::to_string(local_rank) + "_result"));
  }

  rp.deliver(results);
  self->receive([](caf::get_atom) { });
};

void fetch_result_fn(caf::blocking_actor* self,
                     const caf::actor& mpi_actor,
                     const int node_rank,
                     int num_devices_per_node,
                     int batch_id) {
  auto rhs = std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>>();
  for (int local_rank = 0; local_rank < num_devices_per_node; local_rank++) {
    auto result = LoadFromSharedMemory(batch_id, "g" + std::to_string(local_rank) + "_result");
    rhs.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, 0, result, CreateMpiTag(batch_id, TaskType::kFetchResult, node_rank, local_rank)));
  }

  for (int local_rank = 0; local_rank < num_devices_per_node; local_rank++) {
    receive_result<bool>(rhs[local_rank]);
  }

  self->receive([](caf::get_atom) { });
}

}

vertex_cut_executor::vertex_cut_executor(caf::actor_config& config,
                                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                                         caf::strong_actor_ptr mpi_actor_ptr,
                                         int node_rank,
                                         int num_nodes,
                                         int num_devices_per_node,
                                         bool using_precomputed_aggs)
    : executor_actor(config,
                     exec_ctl_actor_ptr,
                     mpi_actor_ptr,
                     node_rank,
                     num_nodes,
                     num_devices_per_node,
                     1),
      using_precomputed_aggs_(using_precomputed_aggs) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  sampler_ = spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, -1);
}

void vertex_cut_executor::Sampling(int batch_id, int) {
  // OPTIMIZATION TODO: Sampling in c++.
  auto sampling_task = spawn(sampling_fn, sampler_, batch_id);
  RequestAndReportTaskDone(sampling_task, TaskType::kSampling, batch_id);
}

void vertex_cut_executor::PrepareInput(int batch_id, int) {
  ReportTaskDone(TaskType::kPrepareInput, batch_id);
}

void vertex_cut_executor::Compute(int batch_id, int) {
  auto compute_task = spawn(gnn_broadcast_execute_fn, gnn_executor_group_, batch_id, gnn_executor_request_type::kComputeRequestType);
  RequestAndReportTaskDone(compute_task, TaskType::kCompute, batch_id);
}

void vertex_cut_executor::PrepareAggregations(int batch_id, int) {
  ReportTaskDone(TaskType::kPrepareAggregations, batch_id);
}

void vertex_cut_executor::RecomputeAggregations(int batch_id, int) {
  ReportTaskDone(TaskType::kRecomputeAggregations, batch_id);
}

void vertex_cut_executor::ComputeRemaining(int batch_id, int) {
  ReportTaskDone(TaskType::kComputeRemaining, batch_id);
}

void vertex_cut_executor::DirectFetchResult(int batch_id, int, caf::response_promise rp) {
  auto task = spawn(direct_fetch_result_fn, num_devices_per_node_, batch_id, rp);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { this->Cleanup(batch_id); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void vertex_cut_executor::FetchResult(int batch_id, int) {
  auto task = spawn(fetch_result_fn, mpi_actor_, node_rank_, num_devices_per_node_, batch_id);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { this->Cleanup(batch_id); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void vertex_cut_executor::Cleanup(int batch_id) {
  object_storages_.erase(batch_id);
}

}
}
