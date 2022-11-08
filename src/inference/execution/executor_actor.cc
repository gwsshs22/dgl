#include "executor_actor.h"

#include "./gnn/gnn_executor.h"
#include "./gnn/graph_server_actor.h"
#include "task_executors.h"

namespace dgl {
namespace inference {

BaseExecutionContext::BaseExecutionContext(int batch_id,
                                   const NDArray& new_ngids,
                                   const NDArray& src_ngids,
                                   const NDArray& dst_ngids)
    : batch_id_(batch_id),
      new_ngids_(new_ngids),
      src_ngids_(src_ngids),
      dst_ngids_(dst_ngids) {
}

BaseExecutionContext::BaseExecutionContext(int batch_id) : batch_id_(batch_id) {
}

void BaseExecutionContext::SetBatchInput(const NDArray& new_ngids,
                                     const NDArray& src_ngids,
                                     const NDArray& dst_ngids) {
  new_ngids_ = new_ngids;
  src_ngids_ = src_ngids;
  dst_ngids_ = dst_ngids;
}

caf::actor spawn_executor_actor(caf::actor_system& system,
                                ParallelizationType parallelization_type,
                                const caf::strong_actor_ptr& exec_ctl_actor_ptr,
                                const caf::strong_actor_ptr& mpi_actor_ptr,
                                int node_rank,
                                int num_nodes,
                                int num_devices_per_node,
                                bool using_precomputed_aggs) {
  if (parallelization_type == ParallelizationType::kData) {
    return system.spawn<data_parallel_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_devices_per_node);
  } else if (parallelization_type == ParallelizationType::kP3) {
    return system.spawn<p3_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_devices_per_node);
  } else {
    return system.spawn<vertex_cut_executor>(
        exec_ctl_actor_ptr, mpi_actor_ptr, node_rank, num_nodes, num_devices_per_node, using_precomputed_aggs);
  }
}

}
}
