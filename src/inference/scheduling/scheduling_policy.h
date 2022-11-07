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

  const bool using_precomputed_aggs_;
  const int num_nodes_;
  const int num_devices_per_node_;

  std::queue<BatchInput> input_queue_;

 private:
  int batch_id_counter_ = 0;
};

struct GpuRepresentation {
  int running_batch_id = -1;
  TaskType running_task_type = TaskType::kNone;
};

struct SamplerRepresentation {
  int running_batch_id = -1;
};

class NodeRepresentation {

 public:
  NodeRepresentation(int num_devices_per_node)
      : gpus_(num_devices_per_node),
        samplers_(num_devices_per_node),
        num_idle_gpus_(num_devices_per_node),
        num_idle_samplers_(num_devices_per_node) {
  }

  inline int num_idle_gpus() {
    return num_idle_gpus_;
  }

  inline int num_idle_samplers() {
    return num_idle_samplers_;
  }

  int assign_sampler(int batch_id) {
    for (int i = 0; i < samplers_.size(); i++) {
      if (samplers_[i].running_batch_id == -1) {
        samplers_[i].running_batch_id = batch_id;
        num_idle_samplers_--;
        return i;
      }
    }

    return -1;
  }

  bool release_sampler(int batch_id) {
    for (int i = 0; i < samplers_.size(); i++) {
      if (samplers_[i].running_batch_id == batch_id) {
        samplers_[i].running_batch_id = -1;
        num_idle_samplers_++;
        return true;
      }
    }

    return false;
  }

  int assign_gpu(int batch_id, TaskType task_type) {
    for (int i = 0; i < gpus_.size(); i++) {
      if (gpus_[i].running_batch_id == -1) {
        gpus_[i].running_batch_id = batch_id;
        gpus_[i].running_task_type = task_type;
        num_idle_gpus_--;
        return i;
      }
    }

    return -1;
  }

  bool release_gpu(int batch_id) {
    for (int i = 0; i < gpus_.size(); i++) {
      if (gpus_[i].running_batch_id == batch_id) {
        gpus_[i].running_batch_id = -1;
        gpus_[i].running_task_type = TaskType::kNone;
        num_idle_gpus_++;
        return true;
      }
    }

    return false;
  }

 private:
  std::vector<GpuRepresentation> gpus_;
  std::vector<SamplerRepresentation> samplers_;
  int num_idle_gpus_;
  int num_idle_samplers_;
};




  static const char* const BatchStatusNames[] = {
    "kInitializing",
    "kInitialized",
    "kSampling",
    "kSampled",
    "kComputing",
    "kFirstLayerComputed", // Only used with cache
    "kComputeRemaining",
    "kComputed",
  };
  
class DataSchedulingPolicy : public BaseSchedulingPolicy {

  enum BatchStatus {
    kInitializing,
    kInitialized,
    kSampling,
    kSampled,
    kComputing,
    kFirstLayerComputed, // Only used with cache
    kComputeRemaining,
    kComputed,
  };



  struct ScheduledBatch {
    int batch_id;
    BatchStatus status = BatchStatus::kInitializing;
    bool input_prepared = false;
    bool input_computing = false;
    bool input_computed = false;
    bool aggregation_prepared = false;
    bool aggregation_recomputing = false;
    bool aggregation_recomputed = false;

    ScheduledBatch(int bid): batch_id(bid) {
    }
  };

 public:
  DataSchedulingPolicy(bool using_precomputed_aggs,
                       int num_nodes,
                       int num_devices_per_node);

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

  std::vector<NodeRepresentation> nodes_;
  std::vector<std::map<int, std::shared_ptr<ScheduledBatch>>> scheduled_batches_;
  std::map<int, int> batch_id_to_global_rank_;
};

class P3SchedulingPolicy : public BaseSchedulingPolicy {

 public:
  P3SchedulingPolicy(bool using_precomputed_aggs,
                     int num_nodes,
                     int num_devices_per_node)
    : BaseSchedulingPolicy(using_precomputed_aggs, num_nodes, num_devices_per_node) {}

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

};

class VertexCutSchedulingPolicy : public BaseSchedulingPolicy {

 public:
  VertexCutSchedulingPolicy(bool using_precomputed_aggs,
                            int num_nodes,
                            int num_devices_per_node)
      : BaseSchedulingPolicy(using_precomputed_aggs, num_nodes, num_devices_per_node) {}

  void OnInitialized(Scheduler& scheduler, int batch_id) override;
  void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) override;

 private:
  void TryScheduling(Scheduler& scheduler) override;

};

}
}
