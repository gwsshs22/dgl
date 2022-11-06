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
                 int node_rank,
                 int num_nodes,
                 int num_devices_per_node);

 private:
  caf::behavior make_behavior() override;
  caf::behavior make_initializing_behavior();
  caf::behavior make_running_behavior();

  void ReportTaskDone(TaskType task_type, int batch_id);

  template <typename T, typename F>
  void RequestAndReportTaskDone(caf::actor& task_executor,
                                TaskType task_type,
                                int batch_id,
                                F&& callback);

  void RequestAndReportTaskDone(caf::actor& task_executor,
                                TaskType task_type,
                                int batch_id);

  // TODO: remote this
  void DoTestTask(int batch_id);

  caf::actor exec_ctl_actor_;
  caf::actor mpi_actor_;
  caf::actor gnn_executor_group_;
  caf::actor graph_server_actor_;

  int num_initialized_components_ = 0;
  int node_rank_;
  int num_nodes_;

  std::map<int, std::shared_ptr<ExecutionContext>> contexts_;
};

}
}
