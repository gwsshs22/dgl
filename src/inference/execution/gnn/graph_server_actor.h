#pragma once

#include <dgl/inference/common.h>

#include "../../process/process_control_actor.h"

namespace dgl {
namespace inference {

class graph_server_actor : public process_control_actor {

 public:
  graph_server_actor(caf::actor_config& config,
                     const caf::strong_actor_ptr& owner_ptr);

 private:
  EnvSetter MakeEnvSetter() override;

  caf::behavior make_running_behavior(const caf::actor& req_handler) override;
};

}
}
