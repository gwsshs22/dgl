#include "process_control_actor.h"

namespace dgl {
namespace inference {

caf::behavior process_monitor_fn(caf::event_based_actor* self) {
  return {
    [&](caf::initialized_atom,
        caf::actor_id owner_process_control_actor_id,
        const caf::strong_actor_ptr& new_actor_ptr) {
      // TODO: error handling
      auto owner_actor_ptr = self->system().registry().get(owner_process_control_actor_id);
      auto owner_actor = caf::actor_cast<caf::actor>(owner_actor_ptr);
      self->send(owner_actor, caf::initialized_atom_v, new_actor_ptr);
    }
  };
}

void process_create_fn(caf::blocking_actor* self, caf::actor& control_actor) {
  return;
}

process_control_actor::process_control_actor(caf::actor_config& config)
    : event_based_actor(config) {
}

caf::behavior process_control_actor::make_behavior() {
  return {
  };
}

}
}
