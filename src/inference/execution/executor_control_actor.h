#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

class executor_control_actor : public caf::event_based_actor {

 public:
  executor_control_actor(caf::actor_config& config,
                         caf::strong_actor_ptr mpi_actor_ptr);

 protected:
  caf::actor scheduler_actor_;
  int num_nodes_ = -1;
  std::vector<caf::actor> executors_;

 private:
  caf::behavior make_behavior() override;

  caf::behavior initializing();
  caf::behavior running();

  void TryRunning();

  std::vector<std::pair<caf::strong_actor_ptr, int>> pending_executors_;
  bool scheduler_connected_ = false;

  caf::actor mpi_actor_;

  std::map<std::pair<TaskType, int>, int> done_task_counter_;
};

}
}
