#include "executor_actor.h"

#include "mem_utils.h"

namespace dgl {
namespace inference {

namespace {


NDArray create_test_result() {
  NDArray test_result = NDArray::FromVector(std::vector<float>({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 }));
  return test_result;
}


void direct_fetch_result_fn(caf::blocking_actor* self,
                            int batch_id,
                            int local_rank,
                            caf::response_promise rp) {
  // auto result = LoadFromSharedMemory(batch_id, "result");
  auto result = create_test_result();
  rp.deliver(result);

  self->receive([](caf::get_atom) { });
};

void fetch_result_fn(caf::blocking_actor* self,
                     const caf::actor& mpi_actor,
                     const int node_rank,
                     int batch_id,
                     int local_rank) {
  // auto result = LoadFromSharedMemory(batch_id, "result");
  auto result = create_test_result();
  auto rh2 = self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, 0, result, CreateMpiTag(batch_id, TaskType::kFetchResult, node_rank));
  receive_result<bool>(rh2);

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
                     0),
      using_precomputed_aggs_(using_precomputed_aggs) {
}

void vertex_cut_executor::Sampling(int batch_id, int local_rank) {
  ReportTaskDone(TaskType::kSampling, batch_id);
}

void vertex_cut_executor::PrepareInput(int batch_id, int local_rank) {
  ReportTaskDone(TaskType::kPrepareInput, batch_id);
}

void vertex_cut_executor::Compute(int batch_id, int local_rank) {
  ReportTaskDone(TaskType::kCompute, batch_id);
}

void vertex_cut_executor::PrepareAggregations(int batch_id, int local_rank) {
  ReportTaskDone(TaskType::kPrepareAggregations, batch_id);
}

void vertex_cut_executor::RecomputeAggregations(int batch_id, int local_rank) {
  ReportTaskDone(TaskType::kRecomputeAggregations, batch_id);
}

void vertex_cut_executor::ComputeRemaining(int batch_id, int local_rank) {
  ReportTaskDone(TaskType::kComputeRemaining, batch_id);
}

void vertex_cut_executor::DirectFetchResult(int batch_id, int local_rank, caf::response_promise rp) {
  auto task = spawn(direct_fetch_result_fn, batch_id, local_rank, rp);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { this->Cleanup(batch_id); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void vertex_cut_executor::FetchResult(int batch_id, int local_rank) {
  auto task = spawn(fetch_result_fn, mpi_actor_, node_rank_, batch_id, local_rank);
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
