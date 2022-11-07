#include "executor_actor.h"

#include "./gnn/gnn_executor.h"
#include "./gnn/graph_server_actor.h"
#include "task_executors.h"

namespace dgl {
namespace inference {

executor_actor::executor_actor(caf::actor_config& config,
                               caf::strong_actor_ptr exec_ctl_actor_ptr,
                               caf::strong_actor_ptr mpi_actor_ptr,
                               int node_rank,
                               int num_nodes,
                               int num_devices_per_node)
    : event_based_actor(config),
      node_rank_(node_rank),
      num_nodes_(num_nodes) {
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
  mpi_actor_ = caf::actor_cast<caf::actor>(mpi_actor_ptr);
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  gnn_executor_group_ = spawn<caf::linked + caf::monitored>(
      gnn_executor_group, self_ptr, num_devices_per_node);
  graph_server_actor_ = spawn<graph_server_actor, caf::linked + caf::monitored>(self_ptr);
}

caf::behavior executor_actor::make_behavior() {
  return make_initializing_behavior();
}

// Initializing
caf::behavior executor_actor::make_initializing_behavior() {
  return {
    [&](caf::initialized_atom, const std::string component_name, int) {
      num_initialized_components_ += 1;
      if (num_initialized_components_ == 2) {
        // gnn_executor_group_ and graph_server_actor_ are initialized.
        send(exec_ctl_actor_, caf::initialized_atom_v, caf::actor_cast<caf::strong_actor_ptr>(this), node_rank_, num_nodes_);
        become(make_running_behavior());
      }
    }
  };
}

// Running
void executor_actor::ReportTaskDone(TaskType task_type, int batch_id) {
  send(exec_ctl_actor_, caf::done_atom_v, task_type, batch_id, node_rank_);
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

inline void executor_actor::RequestAndReportTaskDone(caf::actor& task_executor,
                                              TaskType task_type,
                                              int batch_id) {
  request(task_executor, caf::infinite, caf::get_atom_v).then(
    [=]{ ReportTaskDone(task_type, batch_id); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

void executor_actor::DoTestTask(int batch_id) {
  auto const& context = contexts_[batch_id];
  context->Print();
  if (batch_id == 0) {
    send(gnn_executor_group_, caf::broadcast_atom_v, caf::make_message(batch_id, std::string("String!")));
  } else {
    send(gnn_executor_group_, caf::broadcast_atom_v, caf::make_message(std::string("String!!!!"), batch_id));
  }
}

//
caf::behavior executor_actor::make_running_behavior() {
  return {
    [&](caf::init_atom, int batch_id, const NDArray& new_gnids, const NDArray& src_gnids, const NDArray& dst_gnids) {
      auto context = std::make_shared<ExecutionContext>(batch_id, new_gnids, src_gnids, dst_gnids);
      contexts_.insert(std::make_pair(batch_id, context));
      ReportTaskDone(TaskType::kInitialize, batch_id);
    },
    [&](caf::init_atom, int batch_id) {
      auto context = std::make_shared<ExecutionContext>(batch_id);
      contexts_.insert(std::make_pair(batch_id, context));

      auto receiver = spawn(input_recv_fn, mpi_actor_, CreateMpiTag(batch_id, TaskType::kInitialize));
      RequestAndReportTaskDone<std::vector<NDArray>>(receiver, TaskType::kInitialize, batch_id,
        [=](const std::vector<NDArray> ret) {
          context->SetBatchInput(ret[0], ret[1], ret[2]);
        });
    },
    [&](caf::broadcast_init_atom, int batch_id, const NDArray& new_gnids, const NDArray& src_gnids, const NDArray& dst_gnids) {
      auto context = std::make_shared<ExecutionContext>(batch_id, new_gnids, src_gnids, dst_gnids);
      contexts_.insert(std::make_pair(batch_id, context));
      auto broadcaster = spawn(input_bsend_fn, mpi_actor_, new_gnids, src_gnids, dst_gnids, CreateMpiTag(batch_id, TaskType::kInitialize));
      RequestAndReportTaskDone(broadcaster, TaskType::kInitialize, batch_id);
    },
    [&](caf::broadcast_init_atom, int batch_id) {
      auto context = std::make_shared<ExecutionContext>(batch_id);
      contexts_.insert(std::make_pair(batch_id, context));

      auto receiver = spawn(input_brecv_fn, mpi_actor_, CreateMpiTag(batch_id, TaskType::kInitialize));
      RequestAndReportTaskDone<std::vector<NDArray>>(receiver, TaskType::kInitialize, batch_id,
        [=](const std::vector<NDArray> ret) {
          context->SetBatchInput(ret[0], ret[1], ret[2]);
        });
    },
    // batch task execution
    [&](caf::exec_atom, TaskType task_type, int batch_id, int local_rank) {
      switch (task_type) {
        case TaskType::kSampling:
          // std::cerr << "batch_id=" << batch_id << "kSampling Done" << std::endl;
          ReportTaskDone(TaskType::kSampling, batch_id);
          break;
        case TaskType::kPrepareInput:
          // std::cerr << "batch_id=" << batch_id << "kPrepareInput Done" << std::endl;
          ReportTaskDone(TaskType::kPrepareInput, batch_id);
          break;
        case TaskType::kCompute:
          // std::cerr << "batch_id=" << batch_id << "kCompute Done" << std::endl;
          ReportTaskDone(TaskType::kCompute, batch_id);
          break;
        case TaskType::kPrepareAggregations:
          // std::cerr << "batch_id=" << batch_id << "kPrepareAggregations Done" << std::endl;
          ReportTaskDone(TaskType::kPrepareAggregations, batch_id);
          break;
        case TaskType::kRecomputeAggregations:
          // std::cerr << "batch_id=" << batch_id << "kRecomputeAggregations Done" << std::endl;
          ReportTaskDone(TaskType::kRecomputeAggregations, batch_id);
          break;
        case TaskType::kComputeRemaining:
          // std::cerr << "batch_id=" << batch_id << "kComputeRemaining Done" << std::endl;
          ReportTaskDone(TaskType::kComputeRemaining, batch_id);
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
