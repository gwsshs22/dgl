#include "gnn_executor.h"

namespace dgl {
namespace inference {

gnn_executor::gnn_executor(caf::actor_config& config,
                           const caf::strong_actor_ptr& owner_ptr,
                           int local_rank)
    : process_control_actor(config, owner_ptr, "gnn_executor", local_rank) {
  group_actor_ = caf::actor_cast<caf::actor>(owner_ptr);
}

EnvSetter gnn_executor::MakeEnvSetter() {
  return [=]{
    SetEnv(DGL_INFER_ACTOR_PROCESS_ROLE, role());
    SetEnv(DGL_INFER_LOCAL_RANK, local_rank());
  };
}

caf::behavior gnn_executor::make_running_behavior(const caf::actor& req_handler) {
  return {
    [=](caf::exec_atom, uint64_t req_id, int request_type, int batch_id) {
      send(req_handler, caf::request_atom_v, req_id, request_type, batch_id);
    },
    [=](caf::response_atom, uint64_t req_id) {
      send(group_actor_, caf::done_atom_v, req_id, local_rank());
    }
  };
}

caf::behavior gnn_executor_group(
    caf::stateful_actor<gnn_executor_group_state>* self,
    const caf::strong_actor_ptr& owner_ptr,
    int num_devices_per_node) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(self);
  self->state.num_devices_per_node = num_devices_per_node;
  for (int i = 0; i < num_devices_per_node; i++) {
    auto gnn_executor_actor = self->spawn<gnn_executor, caf::linked + caf::monitored>(self_ptr, i);
    self->state.executors.emplace_back(gnn_executor_actor);
  }

  return {
    [=](caf::initialized_atom, std::string name, int local_rank) {
      self->state.num_initialized++;
      if (self->state.num_initialized == self->state.num_devices_per_node) {
        auto owner_actor = caf::actor_cast<caf::actor>(owner_ptr);
        self->send(owner_actor, caf::initialized_atom_v, "gnn_executor_group", 0);
      }
    },
    [=](caf::broadcast_exec_atom, int batch_id, int request_type) {
      uint64_t req_id = self->state.req_id_counter++;
      auto rp = self->make_response_promise<bool>();
      for (int i = 0; i < num_devices_per_node; i++) {
        self->send(self->state.executors[i], caf::exec_atom_v, req_id, request_type, batch_id);
      }

      self->state.done_req_counter.emplace(std::make_pair(req_id, std::make_pair(num_devices_per_node, rp)));
      return rp;
    },
    [=](caf::exec_atom, int batch_id, int request_type, int local_rank) {
      uint64_t req_id = self->state.req_id_counter++;
      auto rp = self->make_response_promise<bool>();
      self->send(self->state.executors[local_rank], caf::exec_atom_v, req_id, request_type, batch_id);
      self->state.done_req_counter.emplace(std::make_pair(req_id, std::make_pair(1, rp)));
      return rp;
    },
    [=](caf::done_atom, uint64_t req_id, int local_rank) {
      auto it = self->state.done_req_counter.find(req_id);
      ((*it).second.first)--;
      if (it->second.first > 0) {
        return;
      }

      ((*it).second.second).deliver(true);
      self->state.done_req_counter.erase(it);
    }
  };
}

}
}
