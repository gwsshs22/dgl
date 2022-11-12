#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

struct BatchInput {
  int req_id;
  NDArray new_gnids;
  NDArray new_features;
  NDArray src_gnids;
  NDArray dst_gnids;
};

class Scheduler {

 public:
  virtual void LocalInitialize(int batch_id, int node_rank, const BatchInput& batch_input) = 0;

  virtual void LocalExecute(TaskType task_type, int batch_id, int node_rank, int local_rank) = 0;

  virtual void LocalFetchResult(int batch_id, int node_rank, int local_rank) = 0;

  virtual void BroadcastInitialize(int batch_id, const BatchInput& batch_input) = 0;
  
  virtual void BroadcastExecute(TaskType task_type, int batch_id) = 0;

  virtual void BroadcastFetchResult(int batch_id) = 0;

  virtual void ReportResult(int request_id, NDArray result) = 0;

};

class SchedulingPolicy {

 public:
  virtual void OnNewBatch(Scheduler& scheduler, BatchInput&& input) = 0;
  virtual void OnInitialized(Scheduler& scheduler, int batch_id) = 0;
  virtual void OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) = 0;
  virtual void OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) = 0;
};

std::shared_ptr<SchedulingPolicy> CreatePolicy(ParallelizationType type,
                                               bool using_precomputed_aggs,
                                               int num_nodes,
                                               int num_devices_per_node);

}
}
