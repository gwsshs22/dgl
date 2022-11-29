#pragma once

#include <queue>

#include "scheduling.h"

namespace dgl {
namespace inference {

class BaseSchedulingPolicy : public SchedulingPolicy {

 public:
  BaseSchedulingPolicy(int num_nodes,
                       int num_devices_per_node,
                       int num_samplers_per_node,
                       bool execute_one_by_one);

  void OnNewBatch(Scheduler& scheduler,
                  BatchInput&& input) override;

 protected:
  virtual void TryScheduling(Scheduler& scheduler) = 0;

  int IssueBatchId();

  void SetTaskDone(int req_id, TaskType task_type);

  void ReportRequestDone(Scheduler& scheduler, int req_id, const NDArray& result);

  const int num_nodes_;
  const int num_devices_per_node_;
  const int num_samplers_per_node_;
  const bool execute_one_by_one_;

  std::queue<BatchInput> input_queue_;
  std::unordered_map<int, int> batch_id_to_req_id_;

 private:
  int batch_id_counter_ = 0;
  std::unordered_map<int, RequestStats> stats_;
  
};

enum BatchStatus {
  kInitializingStatus,
  kInitializedStatus,
  kSamplingStatus,
  kSampledStatus,
  kPushingComputationGraphStatus,
  kPushedComputationGraphStatus,
  kComputingStatus,
  kComputedStatus,
  kFetchingResultStatus,
  kFinishedStatus
};

struct ScheduledBatch {
  int batch_id;
  int gpu_local_rank;
  int sampler_rank;
  BatchStatus status = BatchStatus::kInitializingStatus;
  ScheduledBatch(int bid): batch_id(bid) {
  }
};

struct MachineStatus {
  int num_allocated_batches = 0;
  int num_initializing = 0;
  int num_sampling = 0;
  int num_push_computation_graph = 0;
  int num_computing = 0;
  std::vector<bool> gpu_running;
  int num_fetching_result = 0;

  MachineStatus(int num_devices_per_node) {
    for (int i = 0; i < num_devices_per_node; i++) {
      gpu_running.push_back(false);
    }
  }

  MachineStatus() {}

  inline int GetIdleGpuIndex() {
    for (int i = 0; i < gpu_running.size(); i++) {
      if (!gpu_running[i]) {
        return i;
      }
    }

    return -1;
  }

  inline void AssignGpu(int gpu_idx) {
    num_computing++;
    CHECK(!gpu_running[gpu_idx]);
    gpu_running[gpu_idx] = true;
  }

  inline void FreeGpu(int gpu_idx) {
    num_computing--;
    CHECK(gpu_running[gpu_idx]);
    gpu_running[gpu_idx] = false;
  }
};

class DataSchedulingPolicy : public BaseSchedulingPolicy {

 public:
  DataSchedulingPolicy(int num_nodes,
                       int num_devices_per_node,
                       int num_samplers_per_node,
                       bool execute_one_by_one);

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;
  void OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

  int max_concurrent_batches_;

  std::vector<std::map<int, std::shared_ptr<ScheduledBatch>>> scheduled_batches_;
  std::map<int, int> batch_id_to_node_rank_;
  std::vector<MachineStatus> machines_;
};

class P3SchedulingPolicy : public BaseSchedulingPolicy {

 public:
  P3SchedulingPolicy(int num_nodes,
                     int num_devices_per_node,
                     int num_samplers_per_node,
                     bool execute_one_by_one);

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;
  void OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

  int max_concurrent_batches_;
  bool compute_running_;
  int latest_compute_done_node_rank_;
  std::vector<std::map<int, std::shared_ptr<ScheduledBatch>>> scheduled_batches_;
  std::map<int, int> batch_id_to_node_rank_;
  std::vector<MachineStatus> machines_;
};

class VertexCutSchedulingPolicy : public BaseSchedulingPolicy {

 public:
  VertexCutSchedulingPolicy(int num_nodes,
                            int num_devices_per_node,
                            int num_samplers_per_node,
                            bool execute_one_by_one,
                            bool using_precomputed_aggs);

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;
  void OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

  bool using_precomputed_aggs_;
  int max_concurrent_batches_;
  int max_sampling_;
  bool compute_running_;
  std::vector<bool> sampler_running_;
  std::map<int, std::shared_ptr<ScheduledBatch>> scheduled_batches_;
  MachineStatus machine_;
};

}
}
