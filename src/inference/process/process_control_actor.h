#pragma once

#include <unordered_map>

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

struct process_monitor_state {
  int global_actor_process_id_counter = 0;
  std::unordered_map<int, caf::strong_actor_ptr> proc_ctrl_actor_map;
};

caf::behavior process_monitor_fn(caf::stateful_actor<process_monitor_state>* self);

void process_creator_fn(caf::blocking_actor* self);

// Called by the master process or the worker process.
void ForkActorProcess(int actor_process_global_id, const EnvSetter& env_setter); 

class process_control_actor : public caf::event_based_actor {

 public:
  process_control_actor(caf::actor_config& config,
                        const caf::strong_actor_ptr& owner_ptr,
                        const std::string& role,
                        int local_rank);

  inline const std::string& role() {
    return role_;
  }

  inline int local_rank() {
    return local_rank_;
  }

 protected:

  virtual EnvSetter MakeEnvSetter() = 0;
  virtual caf::behavior make_running_behavior(const caf::actor& req_handler) = 0;

 private:
  caf::behavior make_behavior();

  caf::strong_actor_ptr owner_ptr_;
  std::string role_;
  int local_rank_;
};

}
}
