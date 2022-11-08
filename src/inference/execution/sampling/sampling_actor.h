#pragma once

#include <dgl/inference/common.h>

#include "../../process/process_control_actor.h"

namespace dgl {
namespace inference {

class sampling_actor : public process_control_actor {

 public:
  sampling_actor(caf::actor_config& config,
                 const caf::strong_actor_ptr& owner_ptr,
                 int local_rank);

 private:
  EnvSetter MakeEnvSetter() override;

  caf::behavior make_running_behavior(const caf::actor& req_handler) override;

  uint64_t req_id_counter_ = 0;
  std::unordered_map<uint64_t, caf::response_promise> rp_map_;
};

}
}
