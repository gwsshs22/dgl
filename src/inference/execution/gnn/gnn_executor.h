#pragma once

#include <dgl/inference/common.h>

#include "../../process/process_control_actor.h"

namespace dgl {
namespace inference {

class gnn_executor : public process_control_actor {

 public:
  gnn_executor(caf::actor_config& config,
               const caf::strong_actor_ptr& owner_ptr,
               int local_rank);

 private:
  EnvSetter MakeEnvSetter() override;

  caf::behavior make_running_behavior(const caf::actor& req_handler) override;

  caf::actor group_actor_;
};

struct gnn_executor_group_state {
  int num_devices_per_node = 0;
  int num_initialized = 0;
  uint64_t req_id_counter = 0;
  std::vector<caf::actor> executors;
  std::map<uint64_t, std::pair<int, caf::response_promise>> done_req_counter;
};

caf::behavior gnn_executor_group(
    caf::stateful_actor<gnn_executor_group_state>* self,
    const caf::strong_actor_ptr& owner_ptr,
    int num_devices_per_node);

}
}
