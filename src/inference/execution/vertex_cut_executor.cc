#include "executor_actor.h"

namespace dgl {
namespace inference {

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

}
}
