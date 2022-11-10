#pragma once

#include <dgl/inference/common.h>

#include "./gnn/gnn_executor.h"
#include "./gnn/graph_server_actor.h"

#include "object_storage_actor.h"
#include "task_executors.h"

namespace dgl {
namespace inference {

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

  virtual void DirectFetchResult(int batch_id, int local_rank, caf::response_promise rp) {
    throw std::runtime_error("DirectFetchResult not implemented");
  }

  virtual void FetchResult(int batch_id, int local_rank) {
    throw std::runtime_error("FetchResult not implemented");
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

  caf::actor exec_ctl_actor_;
  caf::actor mpi_actor_;
  caf::actor gnn_executor_group_;
  caf::actor graph_server_actor_;

  int num_initialized_components_ = 0;
  int node_rank_;
  int num_nodes_;

  std::unordered_map<int, caf::actor> object_storages_;

 private:
  caf::behavior make_behavior() override;
  caf::behavior make_initializing_behavior();
  caf::behavior make_running_behavior();

  int required_init_count_;
};

class data_parallel_executor : public executor_actor {

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

  void DirectFetchResult(int batch_id, int local_rank, caf::response_promise rp) override;

  void FetchResult(int batch_id, int local_rank) override;

  void Cleanup(int batch_id, int local_rank);

  std::vector<caf::actor> samplers_;
};

class p3_executor : public executor_actor {

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

class vertex_cut_executor : public executor_actor {

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
