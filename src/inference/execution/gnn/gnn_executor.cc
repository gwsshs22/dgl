#include "gnn_executor.h"

#include <dgl/inference/envs.h>

namespace dgl {
namespace inference {

gnn_executor::gnn_executor(caf::actor_config& config,
                           const caf::strong_actor_ptr& owner_ptr,
                           int local_rank)
    : process_control_actor(config, owner_ptr,  "gnn_executor", local_rank) {
}

EnvSetter gnn_executor::MakeEnvSetter() {
  return [=]{
    SetEnv(DGL_INFER_ACTOR_PROCESS_ROLE, role());
    SetEnv(DGL_INFER_LOCAL_RANK, local_rank());
  };
}

caf::behavior gnn_executor::make_running_behavior(const caf::strong_actor_ptr& req_handler_ptr) {
  caf::actor req_handler = caf::actor_cast<caf::actor>(req_handler_ptr);
  return {
    [=]() {
    }
  };
}

caf::behavior gnn_executor_group(
    caf::stateful_actor<gnn_executor_group_state>* self,
    const caf::strong_actor_ptr& owner_ptr,
    int num_devices_per_node) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(self);
  self->state.num_devices_per_node = num_devices_per_node;
  for (int i = 0; i < num_devices_per_node; i++) {
    auto gnn_executor_actor = self->spawn<gnn_executor, caf::linked + caf::monitored>(self_ptr, i);
    self->state.executors.emplace_back(gnn_executor_actor);
  }

  return {
    [=](caf::initialized_atom, std::string name, int local_rank) {
      self->state.num_initialized++;
      if (self->state.num_initialized == self->state.num_devices_per_node) {
        auto owner_actor = caf::actor_cast<caf::actor>(owner_ptr);
        self->send(owner_actor, caf::initialized_atom_v);
      }
    }
  };
}

}
}
