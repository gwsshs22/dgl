#include "executor_actor.h"

namespace dgl {
namespace inference {

VertexCutExecutionContext::VertexCutExecutionContext(int batch_id)
    : BaseExecutionContext(batch_id) {
}

VertexCutExecutionContext::VertexCutExecutionContext(int batch_id,
                                                     const NDArray& new_ngids,
                                                     const NDArray& src_ngids,
                                                     const NDArray& dst_ngids)
    : BaseExecutionContext(batch_id, new_ngids, src_ngids, dst_ngids) {
}

vertex_cut_executor::vertex_cut_executor(caf::actor_config& config,
                                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                                         caf::strong_actor_ptr mpi_actor_ptr,
                                         int node_rank,
                                         int num_nodes,
                                         int num_devices_per_node,
                                         bool using_precomputed_aggs)
    : executor_actor<VertexCutExecutionContext>(config,
                                               exec_ctl_actor_ptr,
                                               mpi_actor_ptr,
                                               node_rank,
                                               num_nodes,
                                               num_devices_per_node,
                                               0),
      using_precomputed_aggs_(using_precomputed_aggs) {
}

void vertex_cut_executor::Sampling(int batch_id, int local_rank) {

}

void vertex_cut_executor::PrepareInput(int batch_id, int local_rank) {
  
}

void vertex_cut_executor::Compute(int batch_id, int local_rank) {
  
}

void vertex_cut_executor::PrepareAggregations(int batch_id, int local_rank) {
  
}

void vertex_cut_executor::RecomputeAggregations(int batch_id, int local_rank) {
  
}

void vertex_cut_executor::ComputeRemaining(int batch_id, int local_rank) {
  
}

}
}
