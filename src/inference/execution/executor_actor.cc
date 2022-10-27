#include "executor_actor.h"

namespace dgl {
namespace inference {

executor_actor::executor_actor(caf::actor_config& config,
                               caf::strong_actor_ptr exec_ctl_actor_ptr,
                               int rank,
                               int world_size)
    : event_based_actor(config),
      rank_(rank),
      world_size_(world_size) {
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
}

caf::behavior executor_actor::make_behavior() {
  send(exec_ctl_actor_, caf::initialized_atom_v, caf::actor_cast<caf::strong_actor_ptr>(this), rank_, world_size_);
  return {
    
  };
}

}
}
