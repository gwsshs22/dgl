#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

caf::behavior process_monitor_fn(caf::event_based_actor* self);

void process_create_fn(caf::blocking_actor* self);

class process_control_actor : public caf::event_based_actor {

 public:
  process_control_actor(caf::actor_config& config);

 private:
  caf::behavior make_behavior();
};

}
}
