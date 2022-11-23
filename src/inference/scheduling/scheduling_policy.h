#pragma once

#include <queue>

#include "scheduling.h"

namespace dgl {
namespace inference {

class BaseSchedulingPolicy : public SchedulingPolicy {

 public:
  BaseSchedulingPolicy(bool using_precomputed_aggs,
                       int num_nodes,
                       int num_devices_per_node);

  void OnNewBatch(Scheduler& scheduler,
                  BatchInput&& input) override;

 protected:
  virtual void TryScheduling(Scheduler& scheduler) = 0;

  int IssueBatchId();

  void SetTaskDone(int req_id, TaskType task_type);

  void ReportRequestDone(Scheduler& scheduler, int req_id, const NDArray& result);
  

  const bool using_precomputed_aggs_;
  const int num_nodes_;
  const int num_devices_per_node_;

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
  kComputingStatus,
  kFirstLayerComputedStatus, // Only used with cache
  kComputeRemainingStatus,
  kComputedStatus,
  kResultFetchingStatus,
  kFinishedStatus
};

struct ScheduledBatch {
  int batch_id;
  BatchStatus status = BatchStatus::kInitializingStatus;
  bool input_prepared = false;
  bool input_computing = false;
  bool input_computed = false;
  bool aggregation_prepared = false;
  bool aggregation_recomputing = false;
  bool aggregation_recomputed = false;
  ScheduledBatch(int bid): batch_id(bid) {
  }
};

class DataSchedulingPolicy : public BaseSchedulingPolicy {

 public:
  DataSchedulingPolicy(bool using_precomputed_aggs,
                       int num_nodes,
                       int num_devices_per_node);

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;
  void OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

  std::vector<std::map<int, std::shared_ptr<ScheduledBatch>>> scheduled_batches_;
  std::map<int, int> batch_id_to_global_rank_;
};

class P3SchedulingPolicy : public BaseSchedulingPolicy {

 public:
  P3SchedulingPolicy(bool using_precomputed_aggs,
                     int num_nodes,
                     int num_devices_per_node);

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;
  void OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

  std::vector<std::map<int, std::shared_ptr<ScheduledBatch>>> scheduled_batches_;
  std::map<int, int> batch_id_to_global_rank_;
  bool compute_running_;
};

class VertexCutSchedulingPolicy : public BaseSchedulingPolicy {

 public:
  VertexCutSchedulingPolicy(bool using_precomputed_aggs,
                            int num_nodes,
                            int num_devices_per_node)
      : BaseSchedulingPolicy(using_precomputed_aggs, num_nodes, num_devices_per_node) {}

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;
  void OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

  std::map<int, std::shared_ptr<ScheduledBatch>> scheduled_batches_;
};

}
}
