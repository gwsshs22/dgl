#include "sampling_actor.h"


namespace dgl {
namespace inference {

static const int SAMPLING_REQUEST = 0;
static const int CLEANUP_REQUEST = 1;

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
    [=](caf::sampling_atom, int batch_id) {
      auto rp = make_response_promise<bool>();
      uint64_t req_id = req_id_counter_++;
      send(req_handler, caf::request_atom_v, req_id, SAMPLING_REQUEST, batch_id);
      rp_map_.emplace(std::make_pair(req_id, rp));
      return rp;
    },
    [=](caf::cleanup_atom, int batch_id) {
      auto rp = make_response_promise<bool>();
      uint64_t req_id = req_id_counter_++;
      send(req_handler, caf::request_atom_v, req_id, CLEANUP_REQUEST, batch_id);
      rp_map_.emplace(std::make_pair(req_id, rp));
      return rp;
    },
    [=](caf::response_atom, uint64_t req_id) {
      auto it = rp_map_.find(req_id);
      assert(it != rp_map_.end());
      it->second.deliver(true);
      rp_map_.erase(it);
    }
  };
}

}
}
