#include "obj_store_actor.h"

namespace dgl {
namespace inference {

obj_store_actor::obj_store_actor(caf::actor_config& config)
    : event_based_actor(config) {
}

caf::behavior obj_store_actor::make_behavior() {
  return {

  };
}

}
}
