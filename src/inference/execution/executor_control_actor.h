#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

class executor_control_actor : public caf::event_based_actor {

 public:
  executor_control_actor(caf::actor_config& config);

 private:
  caf::behavior make_behavior() override;

  caf::behavior initializing();
  caf::behavior running();

  int world_size_ = -1;
  std::vector<std::pair<caf::strong_actor_ptr, int>> executors_;
};

}
}
