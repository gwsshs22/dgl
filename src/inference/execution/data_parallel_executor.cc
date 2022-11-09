#include "executor_actor.h"

#include "./sampling/sampling_actor.h"
#include "../process/process_control_actor.h"

namespace dgl {
namespace inference {

namespace {
void prepare_input_fn(caf::blocking_actor* self,
                      const caf::actor& sampler,
                      int batch_id) {
  auto rh = self->request(sampler, caf::infinite, caf::data_parallel_input_fetch_atom_v, batch_id);
  receive_result<bool>(rh);
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
                     num_devices_per_node * 2) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  for (int i = 0; i < num_devices_per_node; i++) {
    samplers_.emplace_back(spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, i));
    input_fetchers_.emplace_back(spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, i));
  }
}

void data_parallel_executor::Sampling(int batch_id, int local_rank) {
  auto sampling_task = spawn(sampling_fn, samplers_[local_rank], batch_id);
  RequestAndReportTaskDone(sampling_task, TaskType::kSampling, batch_id);
}

void data_parallel_executor::PrepareInput(int batch_id, int local_rank) {
  auto prepare_input_task = spawn(prepare_input_fn, input_fetchers_[local_rank], batch_id);
  RequestAndReportTaskDone(prepare_input_task, TaskType::kPrepareInput, batch_id);
}

void data_parallel_executor::Compute(int batch_id, int local_rank) {
  auto compute_task = spawn(gnn_execute_fn, gnn_executor_group_, batch_id, gnn_executor_request_type::kComputeRequestType, local_rank);
  RequestAndReportTaskDone(compute_task, TaskType::kCompute, batch_id);
}

}
}
