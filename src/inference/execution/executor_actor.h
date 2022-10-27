#pragma once

#include <dgl/inference/common.h>

#include "execution_context.h"

namespace dgl {
namespace inference {

class executor_actor : public caf::event_based_actor {

 public:
  executor_actor(caf::actor_config& config,
                 caf::strong_actor_ptr exec_ctl_actor_ptr,
                 caf::strong_actor_ptr mpi_actor_ptr,
                 int rank,
                 int world_size);

 private:
  caf::behavior make_behavior() override;

  void ReportTaskDone(TaskType task_type, int batch_id);

  template <typename T, typename F>
  void RequestAndReportTaskDone(caf::actor& task_executor,
                                TaskType task_type,
                                int batch_id,
                                F&& callback);

  void RequestAndReportTaskDone(caf::actor& task_executor,
                                TaskType task_type,
                                int batch_id);

  void InputBroadcast(int batch_id);
  void InputReceive(int batch_id);

  // TODO: remote this
  void DoTestTask(int batch_id);

  caf::actor exec_ctl_actor_;
  caf::actor mpi_actor_;
  int rank_;
  int world_size_;

  std::map<int, std::shared_ptr<ExecutionContext>> contexts_;
};

}
}
