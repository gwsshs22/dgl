#include "executor_actor.h"

#include "./sampling/sampling_actor.h"

namespace dgl {
namespace inference {

DataParallelExecutionContext::DataParallelExecutionContext(int batch_id)
    : BaseExecutionContext(batch_id) {
}

DataParallelExecutionContext::DataParallelExecutionContext(int batch_id,
                                                           const NDArray& new_ngids,
                                                           const NDArray& src_ngids,
                                                           const NDArray& dst_ngids)
    : BaseExecutionContext(batch_id, new_ngids, src_ngids, dst_ngids) {
}

data_parallel_executor::data_parallel_executor(caf::actor_config& config,
                                               caf::strong_actor_ptr exec_ctl_actor_ptr,
                                               caf::strong_actor_ptr mpi_actor_ptr,
                                               int node_rank,
                                               int num_nodes,
                                               int num_devices_per_node)
    : executor_actor<DataParallelExecutionContext>(config,
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
  std::cerr << "batch_id=" << batch_id << " Sampling Done" << std::endl;
  ReportTaskDone(TaskType::kSampling, batch_id);
}

void data_parallel_executor::PrepareInput(int batch_id, int local_rank) {
  std::cerr << "batch_id=" << batch_id << " PrepareInput Done" << std::endl;
  ReportTaskDone(TaskType::kPrepareInput, batch_id);
}

void data_parallel_executor::Compute(int batch_id, int local_rank) {
  std::cerr << "batch_id=" << batch_id << " Compute Done" << std::endl;
  ReportTaskDone(TaskType::kCompute, batch_id);
}

}
}
