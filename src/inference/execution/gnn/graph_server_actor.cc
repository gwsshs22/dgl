#include "graph_server_actor.h"

namespace dgl {
namespace inference {

graph_server_actor::graph_server_actor(caf::actor_config& config,
                                       const caf::strong_actor_ptr& owner_ptr,
                                       int local_rank)
    : process_control_actor(config, owner_ptr,  "graph_server", local_rank) {
}

EnvSetter graph_server_actor::MakeEnvSetter() {
  return [=]{
    SetEnv(DGL_INFER_ACTOR_PROCESS_ROLE, role());
    SetEnv(DGL_INFER_LOCAL_RANK, local_rank());
  };
}

caf::behavior graph_server_actor::make_running_behavior(const caf::actor& req_handler) {
  return {
    [=](caf::response_atom, uint64_t req_id, const caf::message& msg) {
      // Not expected
    }
  };
}

}
}
