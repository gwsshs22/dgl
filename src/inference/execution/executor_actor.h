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
                 int num_backup_servers,
                 int num_devices_per_node,
                 std::string result_dir,
                 bool collect_stats,
                 int required_init_count);

 protected:

  virtual FeatureSplitMethod GetFeatureSplit(int batch_size, int feature_size) {
    throw std::runtime_error("GetFeatureSplit not implemented");
  }

  virtual void Sampling(int batch_id, int local_rank) {
    throw std::runtime_error("Sampling not implemented");
  }

  virtual void PrepareInput(int batch_id, int local_rank) {
    throw std::runtime_error("PrepareInput not implemented");
  }

  virtual void Compute(int batch_id, int local_rank, int param0, int param1) {
    throw std::runtime_error("Compute not implemented");
  }

  virtual void DirectFetchResult(int batch_id, int local_rank, caf::response_promise rp) {
    throw std::runtime_error("DirectFetchResult not implemented");
  }

  virtual void FetchResult(int batch_id, int local_rank) {
    throw std::runtime_error("FetchResult not implemented");
  }

  virtual void Cleanup(int batch_id, int local_rank) {
    throw std::runtime_error("Cleanup not implemented");
  }

  virtual void WriteExecutorTraces(caf::response_promise rp) {
    throw std::runtime_error("WriteTraces not implemented");
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
  const int node_rank_;
  const int num_nodes_;
  const int num_backup_servers_;
  const int num_devices_per_node_;
  const std::string result_dir_;
  const bool collect_stats_;

  std::unordered_map<int, caf::actor> object_storages_;

 private:
  caf::behavior make_behavior() override;
  caf::behavior make_initializing_behavior();
  caf::behavior make_running_behavior();

  void InputRecvFromSameMachine(int batch_id,
                                const NDArray& new_gnids,
                                const NDArray& new_features,
                                const NDArray& src_gnids,
                                const NDArray& dst_gnids);

  void InputRecv(int batch_id);

  void BroadcastInputSend(BroadcastInitType init_type,
                          int batch_id,
                          const NDArray& new_gnids,
                          const NDArray& new_features,
                          const NDArray& src_gnids,
                          const NDArray& dst_gnids);

  void BroadcastInputRecv(BroadcastInitType init_type, int batch_id);

  int required_init_count_;
};

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

class data_parallel_executor : public executor_actor {

 public:
  data_parallel_executor(caf::actor_config& config,
                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                         caf::strong_actor_ptr mpi_actor_ptr,
                         int node_rank,
                         int num_nodes,
                         int num_backup_servers,
                         int num_devices_per_node,
                         std::string result_dir,
                         bool collect_stats);

 private:
  void Sampling(int batch_id, int local_rank);

  void PrepareInput(int batch_id, int local_rank);

  void Compute(int batch_id, int local_rank, int param0, int param1);

  void DirectFetchResult(int batch_id, int local_rank, caf::response_promise rp) override;

  void FetchResult(int batch_id, int local_rank) override;

  void Cleanup(int batch_id, int local_rank) override;

  void WriteExecutorTraces(caf::response_promise rp) override;

  std::vector<caf::actor> samplers_;
};

class p3_executor : public executor_actor {

 public:
  p3_executor(caf::actor_config& config,
              caf::strong_actor_ptr exec_ctl_actor_ptr,
              caf::strong_actor_ptr mpi_actor_ptr,
              int node_rank,
              int num_nodes,
              int num_backup_servers,
              int num_devices_per_node,
              std::string result_dir,
              bool collect_stats);

 private:
  FeatureSplitMethod GetFeatureSplit(int batch_size, int feature_size) override;

  void Sampling(int batch_id, int local_rank) override;

  void PrepareInput(int batch_id, int local_rank) override;

  void Compute(int batch_id, int local_rank, int param0, int param1) override;

  void DirectFetchResult(int batch_id, int local_rank, caf::response_promise rp) override;

  void FetchResult(int batch_id, int local_rank) override;

  void Cleanup(int batch_id, int local_rank) override;

  void WriteExecutorTraces(caf::response_promise rp) override;

  std::vector<caf::actor> samplers_;
  std::unordered_map<int, int> assigned_local_rank_;
};

class vertex_cut_executor : public executor_actor {

 public:
  vertex_cut_executor(caf::actor_config& config,
                      caf::strong_actor_ptr exec_ctl_actor_ptr,
                      caf::strong_actor_ptr mpi_actor_ptr,
                      int node_rank,
                      int num_nodes,
                      int num_backup_servers,
                      int num_devices_per_node,
                      std::string result_dir,
                      bool collect_stats,
                      bool using_precomputed_aggs);

 private:
  FeatureSplitMethod GetFeatureSplit(int batch_size, int feature_size) override;

  void Sampling(int batch_id, int local_rank);

  void PrepareInput(int batch_id, int local_rank);

  void Compute(int batch_id, int local_rank, int param0, int param1);

  void DirectFetchResult(int batch_id, int local_rank, caf::response_promise rp) override;

  void FetchResult(int batch_id, int local_rank) override;

  void Cleanup(int batch_id, int) override;

  void WriteExecutorTraces(caf::response_promise rp) override;

  const bool using_precomputed_aggs_;

  caf::actor sampler_;
};

caf::actor spawn_executor_actor(caf::actor_system& system,
                                ParallelizationType parallelization_type,
                                const caf::strong_actor_ptr& exec_ctl_actor_ptr,
                                const caf::strong_actor_ptr& mpi_actor_ptr,
                                int node_rank,
                                int num_nodes,
                                int num_backup_servers,
                                int num_devices_per_node,
                                std::string result_dir,
                                bool collect_stats,
                                bool using_precomputed_aggs);

}
}
