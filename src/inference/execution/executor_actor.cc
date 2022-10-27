#include "executor_actor.h"

#include "task_executors.h"

namespace dgl {
namespace inference {

executor_actor::executor_actor(caf::actor_config& config,
                               caf::strong_actor_ptr exec_ctl_actor_ptr,
                               caf::strong_actor_ptr mpi_actor_ptr,
                               int rank,
                               int world_size)
    : event_based_actor(config),
      rank_(rank),
      world_size_(world_size) {
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
  mpi_actor_ = caf::actor_cast<caf::actor>(mpi_actor_ptr);
}

void executor_actor::ReportTaskDone(TaskType task_type, int batch_id) {
  send(exec_ctl_actor_, caf::done_atom_v, task_type, batch_id, rank_);
}

template <typename T, typename F>
void executor_actor::RequestAndReportTaskDone(caf::actor& task_executor,
                                              TaskType task_type,
                                              int batch_id,
                                              F&& callback) {
  request(task_executor, caf::infinite, caf::get_atom_v).then(
    [=](const T& ret) {
      callback(ret);
      ReportTaskDone(task_type, batch_id);
    },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void executor_actor::RequestAndReportTaskDone(caf::actor& task_executor,
                                              TaskType task_type,
                                              int batch_id) {
  request(task_executor, caf::infinite, caf::get_atom_v).then(
    [=]{ ReportTaskDone(task_type, batch_id); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void executor_actor::InputBroadcast(int batch_id) {
  auto const& context = contexts_[batch_id];
  auto broadcaster = system().spawn(
    input_broadcast_actor, mpi_actor_, context->new_ngids(), context->src_ngids(), context->dst_ngids());
  RequestAndReportTaskDone(broadcaster, TaskType::kInputBroadcast, batch_id);
}

void executor_actor::InputReceive(int batch_id) {
  auto receiver = system().spawn(input_receive_actor, mpi_actor_);
  auto& context = contexts_[batch_id];
  RequestAndReportTaskDone<std::vector<NDArray>>(receiver, TaskType::kInputBroadcast, batch_id,
    [=](const std::vector<NDArray> ret) {
      context->SetBatchInput(ret[0], ret[1], ret[2]);
    });
}

void executor_actor::DoTestTask(int batch_id) {
  auto const& context = contexts_[batch_id];
  context->Print();
}

// 

caf::behavior executor_actor::make_behavior() {
  send(exec_ctl_actor_, caf::initialized_atom_v, caf::actor_cast<caf::strong_actor_ptr>(this), rank_, world_size_);
  return {
    [&](caf::init_atom, int batch_id, const NDArray& new_gnids, const NDArray& src_gnids, const NDArray& dst_gnids) {
      contexts_.insert(std::make_pair(batch_id, std::make_shared<ExecutionContext>(batch_id, new_gnids, src_gnids, dst_gnids)));
      ReportTaskDone(TaskType::kInitialize, batch_id);
    },
    [&](caf::init_atom, int batch_id) {
      contexts_.insert(std::make_pair(batch_id, std::make_shared<ExecutionContext>(batch_id)));
      ReportTaskDone(TaskType::kInitialize, batch_id);
    },
    [&](caf::exec_atom, TaskType task_type, int batch_id) {
      switch (task_type) {
        case TaskType::kInputBroadcast:
          if (rank_ == 0) {
            InputBroadcast(batch_id);
          } else {
            InputReceive(batch_id);
          }
          break;
        case TaskType::kTest:
          DoTestTask(batch_id);
          break;
      }
    }
  };
}

}
}
