#include "executor_actor.h"

namespace dgl {
namespace inference {

P3ExecutionContext::P3ExecutionContext(int batch_id)
    : BaseExecutionContext(batch_id) {
}

P3ExecutionContext::P3ExecutionContext(int batch_id,
                                       const NDArray& new_ngids,
                                       const NDArray& src_ngids,
                                       const NDArray& dst_ngids)
    : BaseExecutionContext(batch_id, new_ngids, src_ngids, dst_ngids) {
}

p3_executor::p3_executor(caf::actor_config& config,
                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                         caf::strong_actor_ptr mpi_actor_ptr,
                         int node_rank,
                         int num_nodes,
                         int num_devices_per_node)
    : executor_actor<P3ExecutionContext>(config,
                                         exec_ctl_actor_ptr,
                                         mpi_actor_ptr,
                                         node_rank,
                                         num_nodes,
                                         num_devices_per_node,
                                         0) {
  // gnn_executor_group_ = spawn<caf::linked + caf::monitored>(
  //       gnn_executor_group, self_ptr, num_devices_per_node);
}

void p3_executor::Sampling(int batch_id, int local_rank) {

}

void p3_executor::PrepareInput(int batch_id, int local_rank) {
  
}

void p3_executor::Compute(int batch_id, int local_rank) {
  
}

}
}
