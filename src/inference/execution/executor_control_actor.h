#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

caf::actor spawn_executor_control_actor(caf::actor_system& system, SchedulingType scheduling_type);

class executor_control_actor : public caf::event_based_actor {

 public:
  executor_control_actor(caf::actor_config& config);

 protected:
  caf::actor scheduler_actor_;
  int num_nodes_ = -1;
  std::vector<caf::actor> executors_;

 private:
  caf::behavior make_behavior() override;

  caf::behavior initializing();
  virtual caf::behavior running() = 0;

  void TryRunning();

  std::vector<std::pair<caf::strong_actor_ptr, int>> pending_executors_;
  bool scheduler_connected_ = false;
};

class gang_scheduling_executor_control_actor : public executor_control_actor {

 public:
  gang_scheduling_executor_control_actor(caf::actor_config& config);

 private:
  caf::behavior running() override;

  std::map<std::pair<TaskType, int>, int> done_task_counter_;
};

class independent_scheduling_executor_control_actor : public executor_control_actor {

 public:
  independent_scheduling_executor_control_actor(caf::actor_config& config);

 private:
  caf::behavior running() override;

};

}
}
