#include "scheduler_actor.h"

namespace dgl {
namespace inference {

scheduler_actor::scheduler_actor(caf::actor_config& config,
                                const caf::strong_actor_ptr& exec_ctl_actor_ptr,
                                ParallelizationType parallelization_type,
                                bool using_precomputed_aggs,
                                int num_nodes,
                                int num_devices_per_node)
    : event_based_actor(config) {
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
  policy_ = CreatePolicy(parallelization_type, using_precomputed_aggs, num_nodes, num_devices_per_node);
}

void scheduler_actor::LocalInitialize(int batch_id, int node_rank, const BatchInput& batch_input) {
  send(exec_ctl_actor_, caf::init_atom_v, batch_id, node_rank,
       batch_input.new_gnids, batch_input.src_gnids, batch_input.dst_gnids);
}

void scheduler_actor::LocalExecute(TaskType task_type, int batch_id, int node_rank, int local_rank) {
  send(exec_ctl_actor_, caf::exec_atom_v, task_type, batch_id, node_rank, local_rank);
}

void scheduler_actor::BroadcastInitialize(int batch_id, const BatchInput& batch_input) {
  send(exec_ctl_actor_, caf::broadcast_init_atom_v, batch_id, 
       batch_input.new_gnids, batch_input.src_gnids, batch_input.dst_gnids);
}

void scheduler_actor::BroadcastExecute(TaskType task_type, int batch_id) {
  send(exec_ctl_actor_, caf::broadcast_exec_atom_v, task_type, batch_id);
}

caf::behavior scheduler_actor::make_behavior() {
  send(exec_ctl_actor_, caf::set_atom_v);
  return {
    [&](caf::enqueue_atom,
        const NDArray& new_gnids,
        const NDArray& src_gnids,
        const NDArray& dst_gnids) {
      policy_->OnNewBatch(*this, BatchInput { new_gnids, src_gnids, dst_gnids });
    },
    [&](caf::initialized_atom, int batch_id) {
      policy_->OnInitialized(*this, batch_id);
    },
    [&](caf::done_atom, TaskType task_type, int batch_id) {
      policy_->OnExecuted(*this, batch_id, task_type);
    }
  };
}

}
}
