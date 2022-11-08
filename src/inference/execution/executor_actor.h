#pragma once

#include <dgl/inference/common.h>

#include "./gnn/gnn_executor.h"
#include "./gnn/graph_server_actor.h"

#include "task_executors.h"

namespace dgl {
namespace inference {

class BaseExecutionContext {

 public:
  BaseExecutionContext(int batch_id);

  BaseExecutionContext(int batch_id, const NDArray& new_ngids, const NDArray& src_ngids, const NDArray& dst_ngids);

  BaseExecutionContext(const BaseExecutionContext& other) = delete;
  BaseExecutionContext(BaseExecutionContext&& other) = delete;

  void SetBatchInput(const NDArray& new_ngids,
                     const NDArray& src_ngids,
                     const NDArray& dst_ngids);

  inline const NDArray& new_ngids() {
    return new_ngids_;
  }

  inline const NDArray& src_ngids() {
    return src_ngids_;
  }

  inline const NDArray& dst_ngids() {
    return dst_ngids_;
  }

  // TODO: remove this
  void Print() {
    std::cerr << "batch_id = " << batch_id_ << ", " << new_ngids_ << ", " << src_ngids_ << ", " << dst_ngids_ << std::endl;
  }

 private:
  const int batch_id_;

  NDArray new_ngids_;
  NDArray src_ngids_;
  NDArray dst_ngids_;
};

template <typename ExecutionContext>
class executor_actor : public caf::event_based_actor {

 public:
  executor_actor(caf::actor_config& config,
                 caf::strong_actor_ptr exec_ctl_actor_ptr,
                 caf::strong_actor_ptr mpi_actor_ptr,
                 int node_rank,
                 int num_nodes,
                 int num_devices_per_node,
                 int required_init_count);

 protected:
  virtual void Sampling(int batch_id, int local_rank) {
    throw std::runtime_error("Sampling not implemented");
  }

  virtual void PrepareInput(int batch_id, int local_rank) {
    throw std::runtime_error("PrepareInput not implemented");
  }

  virtual void Compute(int batch_id, int local_rank) {
    throw std::runtime_error("Compute not implemented");
  }

  virtual void PrepareAggregations(int batch_id, int local_rank) {
    throw std::runtime_error("PrepareAggregations not implemented");
  }

  virtual void RecomputeAggregations(int batch_id, int local_rank) {
    throw std::runtime_error("RecomputeAggregations not implemented");
  }

  virtual void ComputeRemaining(int batch_id, int local_rank) {
    throw std::runtime_error("ComputeRemaining not implemented");
  }

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

 private:
  caf::behavior make_behavior() override;
  caf::behavior make_initializing_behavior();
  caf::behavior make_running_behavior();

  int required_init_count_;
};

template <typename ExecutionContext>
executor_actor<ExecutionContext>::executor_actor(caf::actor_config& config,
                                                 caf::strong_actor_ptr exec_ctl_actor_ptr,
                                                 caf::strong_actor_ptr mpi_actor_ptr,
                                                 int node_rank,
                                                 int num_nodes,
                                                 int num_devices_per_node,
                                                 int required_init_count)
    : event_based_actor(config),
      node_rank_(node_rank),
      num_nodes_(num_nodes),
      required_init_count_(required_init_count + 2) { // Include the common graph_server_actor and gnn_executor_group
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
  mpi_actor_ = caf::actor_cast<caf::actor>(mpi_actor_ptr);
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  gnn_executor_group_ = spawn<caf::linked + caf::monitored>(
        gnn_executor_group, self_ptr, num_devices_per_node);
  graph_server_actor_ = spawn<graph_server_actor, caf::linked + caf::monitored>(self_ptr);
}

template <typename ExecutionContext>
caf::behavior executor_actor<ExecutionContext>::make_behavior() {
  return make_initializing_behavior();
}

// Initializing
template <typename ExecutionContext>
caf::behavior executor_actor<ExecutionContext>::make_initializing_behavior() {
  return {
    [&](caf::initialized_atom, const std::string component_name, int) {
      num_initialized_components_ += 1;
      if (num_initialized_components_ == required_init_count_) {
        // All components are initialized.
        send(exec_ctl_actor_, caf::initialized_atom_v, caf::actor_cast<caf::strong_actor_ptr>(this), node_rank_, num_nodes_);
        become(make_running_behavior());
      }
    }
  };
}

// Running
template <typename ExecutionContext>
void executor_actor<ExecutionContext>::ReportTaskDone(TaskType task_type, int batch_id) {
  send(exec_ctl_actor_, caf::done_atom_v, task_type, batch_id, node_rank_);
}

template <typename ExecutionContext>
template <typename T, typename F>
void executor_actor<ExecutionContext>::RequestAndReportTaskDone(caf::actor& task_executor,
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

template <typename ExecutionContext>
inline void executor_actor<ExecutionContext>::RequestAndReportTaskDone(caf::actor& task_executor,
                                              TaskType task_type,
                                              int batch_id) {
  request(task_executor, caf::infinite, caf::get_atom_v).then(
    [=]{ ReportTaskDone(task_type, batch_id); },
    [&](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << caf::to_string(err) << std::endl;
    });
}

template <typename ExecutionContext>
void executor_actor<ExecutionContext>::DoTestTask(int batch_id) {
  auto const& context = contexts_[batch_id];
  context->Print();
  if (batch_id == 0) {
    send(gnn_executor_group_, caf::broadcast_atom_v, caf::make_message(batch_id, std::string("String!")));
  } else {
    send(gnn_executor_group_, caf::broadcast_atom_v, caf::make_message(std::string("String!!!!"), batch_id));
  }
}

//
template <typename ExecutionContext>
caf::behavior executor_actor<ExecutionContext>::make_running_behavior() {
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
          Sampling(batch_id, local_rank);
          break;
        case TaskType::kPrepareInput:
          PrepareInput(batch_id, local_rank);
          break;
        case TaskType::kCompute:
          Compute(batch_id, local_rank);
          break;
        case TaskType::kPrepareAggregations:
          PrepareAggregations(batch_id, local_rank);
          break;
        case TaskType::kRecomputeAggregations:
          RecomputeAggregations(batch_id, local_rank);
          break;
        case TaskType::kComputeRemaining:
          ComputeRemaining(batch_id, local_rank);
          break;
        case TaskType::kTest:
          DoTestTask(batch_id);
          break;
      }
    }
  };
}

class DataParallelExecutionContext : public BaseExecutionContext {

 public:
  DataParallelExecutionContext(int batch_id);

  DataParallelExecutionContext(int batch_id, const NDArray& new_ngids, const NDArray& src_ngids, const NDArray& dst_ngids);

};

class data_parallel_executor : public executor_actor<DataParallelExecutionContext> {

 public:
  data_parallel_executor(caf::actor_config& config,
                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                         caf::strong_actor_ptr mpi_actor_ptr,
                         int node_rank,
                         int num_nodes,
                         int num_devices_per_node);

 private:
  void Sampling(int batch_id, int local_rank);

  void PrepareInput(int batch_id, int local_rank);

  void Compute(int batch_id, int local_rank);

  std::vector<caf::actor> samplers_;
};

class P3ExecutionContext : public BaseExecutionContext {

 public:
  P3ExecutionContext(int batch_id);

  P3ExecutionContext(int batch_id, const NDArray& new_ngids, const NDArray& src_ngids, const NDArray& dst_ngids);

};

class p3_executor : public executor_actor<P3ExecutionContext> {

 public:
  p3_executor(caf::actor_config& config,
              caf::strong_actor_ptr exec_ctl_actor_ptr,
              caf::strong_actor_ptr mpi_actor_ptr,
              int node_rank,
              int num_nodes,
              int num_devices_per_node);

 private:
  void Sampling(int batch_id, int local_rank);

  void PrepareInput(int batch_id, int local_rank);

  void Compute(int batch_id, int local_rank);

};

class VertexCutExecutionContext : public BaseExecutionContext {

 public:
  VertexCutExecutionContext(int batch_id);

  VertexCutExecutionContext(int batch_id, const NDArray& new_ngids, const NDArray& src_ngids, const NDArray& dst_ngids);

};

class vertex_cut_executor : public executor_actor<VertexCutExecutionContext> {

 public:
  vertex_cut_executor(caf::actor_config& config,
                      caf::strong_actor_ptr exec_ctl_actor_ptr,
                      caf::strong_actor_ptr mpi_actor_ptr,
                      int node_rank,
                      int num_nodes,
                      int num_devices_per_node,
                      bool using_precomputed_aggs);

 private:
  void Sampling(int batch_id, int local_rank);

  void PrepareInput(int batch_id, int local_rank);

  void Compute(int batch_id, int local_rank);

  void PrepareAggregations(int batch_id, int local_rank);

  void RecomputeAggregations(int batch_id, int local_rank);

  void ComputeRemaining(int batch_id, int local_rank);

  const bool using_precomputed_aggs_;
};

caf::actor spawn_executor_actor(caf::actor_system& system,
                                ParallelizationType parallelization_type,
                                const caf::strong_actor_ptr& exec_ctl_actor_ptr,
                                const caf::strong_actor_ptr& mpi_actor_ptr,
                                int node_rank,
                                int num_nodes,
                                int num_devices_per_node,
                                bool using_precomputed_aggs);

}
}
