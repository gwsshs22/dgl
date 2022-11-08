#include "sampling_actor.h"


namespace dgl {
namespace inference {

sampling_actor::sampling_actor(caf::actor_config& config,
                               const caf::strong_actor_ptr& owner_ptr,
                               int local_rank)
    : process_control_actor(config, owner_ptr,  "sampler", local_rank) {
}

EnvSetter sampling_actor::MakeEnvSetter() {
  return [=]{
    SetEnv(DGL_INFER_ACTOR_PROCESS_ROLE, role());
    SetEnv(DGL_INFER_LOCAL_RANK, local_rank());
  };
}

caf::behavior sampling_actor::make_running_behavior(const caf::actor& req_handler) {
  return {
    [=](caf::exec_atom, const caf::message& msg) {
      send(req_handler, caf::request_atom_v, req_id_counter_++, msg);
    },
    [=](caf::response_atom, uint64_t req_id, const caf::message& msg) {
      std::cerr << "sampler(" << local_rank() << ") returns "
        << "(req_id=" << req_id << ")" << caf::to_string(msg) << std::endl;
    }
  };
}

}
}
