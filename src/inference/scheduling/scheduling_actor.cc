#include "scheduling_actor.h"

namespace dgl {
namespace inference {

scheduling_actor::scheduling_actor(caf::actor_config& config,
                                   const caf::strong_actor_ptr& exec_ctl_actor_ptr)
    : event_based_actor(config),
      exec_ctl_actor_ptr_(exec_ctl_actor_ptr) {
}

caf::behavior scheduling_actor::make_behavior() {
  return {
    [&](caf::enqueue_atom,
        const NDArray& new_node_orig_ids,
        const NDArray& src_orig_ids,
        const NDArray& dst_orig_ids) {

      TrySchedule();
    }
  };
}

void scheduling_actor::TrySchedule() {

}

}
}
