#pragma once

#include <dgl/inference/actor_types.h>

namespace dgl {
namespace inference {

class executor_actor : public caf::event_based_actor {

 public:
  executor_actor(caf::actor_config& config,
                 caf::strong_actor_ptr exec_ctl_actor_ptr,
                 int rank,
                 int world_size);

 private:
  caf::behavior make_behavior() override;

  caf::strong_actor_ptr exec_ctl_actor_ptr_;
  int rank_;
  int world_size_;
};

}
}
