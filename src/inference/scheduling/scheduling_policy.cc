#include "scheduling_policy.h"

#include <limits>

namespace dgl {
namespace inference {

std::shared_ptr<SchedulingPolicy> CreatePolicy(ParallelizationType type,
                                               bool using_precomputed_aggs,
                                               int num_nodes,
                                               int num_devices_per_node) {
  bool using_precomputed_aggs_tmp = false; // Revisit here after implementing P^3 and vcut with precomputed aggrs.
  if (type == ParallelizationType::kData) {
    return std::make_shared<DataSchedulingPolicy>(using_precomputed_aggs_tmp, num_nodes, num_devices_per_node);
  } else if (type == ParallelizationType::kP3) {
    return std::make_shared<P3SchedulingPolicy>(using_precomputed_aggs_tmp, num_nodes, num_devices_per_node);
  } else {
    return std::make_shared<VertexCutSchedulingPolicy>(using_precomputed_aggs_tmp, num_nodes, num_devices_per_node);
  }
}

BaseSchedulingPolicy::BaseSchedulingPolicy(bool using_precomputed_aggs,
                                           int num_nodes,
                                           int num_devices_per_node)
    : using_precomputed_aggs_(using_precomputed_aggs),
      num_nodes_(num_nodes),
      num_devices_per_node_(num_devices_per_node) {
}

void BaseSchedulingPolicy::OnNewBatch(Scheduler& scheduler, 
                                      BatchInput&& input) {
  input_queue_.push(std::move(input));
  TryScheduling(scheduler);
}

int BaseSchedulingPolicy::IssueBatchId() {
  return batch_id_counter_++;
}

//////////////////////////
// DataSchedulingPolicy //
//////////////////////////
DataSchedulingPolicy::DataSchedulingPolicy(bool using_precomputed_aggs,
                                           int num_nodes,
                                           int num_devices_per_node)
   : BaseSchedulingPolicy(using_precomputed_aggs, num_nodes, num_devices_per_node) {
  
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_devices_per_node; j++) {
      scheduled_batches_.emplace_back(std::map<int, std::shared_ptr<ScheduledBatch>>());
    }
  }
}

void DataSchedulingPolicy::TryScheduling(Scheduler& scheduler) {
  while (!input_queue_.empty()) {
    int min_allocated_num = std::numeric_limits<int>::max();
    int global_rank = -1;

    for (int i = 0; i < num_nodes_ * num_devices_per_node_; i++) {
      if (scheduled_batches_[i].size() < min_allocated_num) {
        min_allocated_num = scheduled_batches_[i].size();
        global_rank = i;
      }
    }

    if (min_allocated_num >= 8) {
      break;
    }

    int batch_id = IssueBatchId();
    int node_rank = global_rank / num_devices_per_node_;
    auto front = input_queue_.front();
    batch_id_to_req_id_[batch_id] = front.req_id;
    scheduler.LocalInitialize(batch_id, node_rank, front);
    input_queue_.pop();
 
    scheduled_batches_[global_rank].emplace(std::make_pair(batch_id, std::make_shared<ScheduledBatch>(batch_id)));
    batch_id_to_global_rank_[batch_id] = global_rank;
  }

  for (int global_rank = 0; global_rank < num_nodes_ * num_devices_per_node_; global_rank++) {
    int node_rank = global_rank / num_devices_per_node_;
    int local_rank = global_rank % num_devices_per_node_;
    auto& scheduled_batches = scheduled_batches_[global_rank];

    bool is_first_batch = true;

    for (auto it = scheduled_batches.begin(); it != scheduled_batches.end();) {
      int batch_id = it->first;
      auto& scheduled_batch = it->second;
      auto batch_finished = false;

      if (scheduled_batch->status == BatchStatus::kInitializedStatus) {
        scheduled_batch->status = BatchStatus::kSamplingStatus;
        scheduler.LocalExecute(TaskType::kSampling, batch_id, node_rank, local_rank);
      } else if (scheduled_batch->status == BatchStatus::kSampledStatus) {
        scheduled_batch->status = BatchStatus::kComputingStatus;

        scheduler.LocalExecute(TaskType::kPrepareInput, batch_id, node_rank, local_rank);
        if (using_precomputed_aggs_) { 
          scheduler.LocalExecute(TaskType::kPrepareAggregations, batch_id, node_rank, local_rank);
        }
      } else if (scheduled_batch->status == BatchStatus::kComputingStatus && is_first_batch) { // Only the first batch can proceed computation
        if (scheduled_batch->input_prepared && !scheduled_batch->input_computing) {
          scheduled_batch->input_computing = true;
          scheduler.LocalExecute(TaskType::kCompute, batch_id, node_rank, local_rank);
        }

        if (using_precomputed_aggs_ && scheduled_batch->aggregation_prepared && !scheduled_batch->aggregation_recomputing) {
          scheduled_batch->aggregation_recomputing = true;
          scheduler.LocalExecute(TaskType::kRecomputeAggregations, batch_id, node_rank, local_rank);
        }
      } else if (scheduled_batch->status == BatchStatus::kFirstLayerComputedStatus) { // Only for precomputed aggregations
        scheduled_batch->status = BatchStatus::kComputeRemainingStatus;
        scheduler.LocalExecute(TaskType::kComputeRemaining, batch_id, node_rank, local_rank);
      } else if (scheduled_batch->status == BatchStatus::kComputedStatus) {
        scheduled_batch->status = BatchStatus::kResultFetchingStatus;
        scheduler.LocalFetchResult(batch_id, node_rank, local_rank);
      } else if (scheduled_batch->status == BatchStatus::kFinishedStatus) {
        batch_finished = true;
      }

      if (batch_finished) {
        scheduled_batches.erase(it++);
      } else {
        ++it;
        is_first_batch = false;
      }
    }
  }
}

void DataSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  auto global_rank = batch_id_to_global_rank_[batch_id];
  auto it = scheduled_batches_[global_rank].find(batch_id);
  auto& scheduled_batch = it->second;

  if (task_type == TaskType::kSampling) {
    scheduled_batch->status = kSampledStatus;
    TryScheduling(scheduler);
    return;
  }

  if (task_type == TaskType::kPrepareInput) {
    scheduled_batch->input_prepared = true;
  } else if (task_type == TaskType::kCompute) {
    assert(scheduled_batch->input_prepared && scheduled_batch->input_computing);
    scheduled_batch->input_computed = true;
  } else if (task_type == TaskType::kPrepareAggregations) {
    scheduled_batch->aggregation_prepared = true;
  } else if (task_type == TaskType::kRecomputeAggregations) {
    assert(scheduled_batch->aggregation_prepared && scheduled_batch->aggregation_recomputing);
    scheduled_batch->aggregation_recomputed = true;
  } else if (task_type == TaskType::kComputeRemaining) {
    assert(using_precomputed_aggs_);
    scheduled_batch->status = BatchStatus::kComputedStatus;
  }

  if (using_precomputed_aggs_) {
    if (scheduled_batch->input_computed && scheduled_batch->aggregation_recomputed &&
        scheduled_batch->status == BatchStatus::kComputingStatus) {
      scheduled_batch->status = BatchStatus::kFirstLayerComputedStatus;
    }
  } else {
    if (scheduled_batch->input_computed) {
      scheduled_batch->status = BatchStatus::kComputedStatus;
    }
  }

  TryScheduling(scheduler);
}

void DataSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  auto global_rank = batch_id_to_global_rank_[batch_id];
  auto it = scheduled_batches_[global_rank].find(batch_id);
  assert(it != scheduled_batches_[global_rank].end());
  assert(it->second->status == kInitializingStatus);
  it->second->status = kInitializedStatus;

  TryScheduling(scheduler);
}

void DataSchedulingPolicy::OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) {
  auto req_id = batch_id_to_req_id_[batch_id];
  scheduler.ReportResult(req_id, result);

  auto global_rank = batch_id_to_global_rank_[batch_id];
  auto it = scheduled_batches_[global_rank].find(batch_id);
  assert(it != scheduled_batches_[global_rank].end());
  assert(it->second->status == kResultFetchingStatus);
  it->second->status = kFinishedStatus;

  batch_id_to_global_rank_.erase(batch_id);
  batch_id_to_req_id_.erase(batch_id);

  TryScheduling(scheduler);
}

////////////////////////
// P3SchedulingPolicy //
////////////////////////
void P3SchedulingPolicy::TryScheduling(Scheduler& scheduler) { 

}

void P3SchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  TryScheduling(scheduler);
}

void P3SchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  
}

void P3SchedulingPolicy::OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) {
  
}

///////////////////////////////
// VertexCutSchedulingPolicy //
///////////////////////////////
void VertexCutSchedulingPolicy::TryScheduling(Scheduler& scheduler) {
  while (!input_queue_.empty()) {
    if (scheduled_batches_.size() >= 8) {
      break;
    }

    int batch_id = IssueBatchId();
    auto front = input_queue_.front();
    batch_id_to_req_id_[batch_id] = front.req_id;
    scheduler.BroadcastInitialize(batch_id, front);
    input_queue_.pop();
 
    scheduled_batches_.emplace(std::make_pair(batch_id, std::make_shared<ScheduledBatch>(batch_id)));
  }

  bool is_first_batch = true;

  for (auto it = scheduled_batches_.begin(); it != scheduled_batches_.end();) {
    int batch_id = it->first;
    auto& scheduled_batch = it->second;
    auto batch_finished = false;

    if (scheduled_batch->status == BatchStatus::kInitializedStatus) {
      scheduled_batch->status = BatchStatus::kSamplingStatus;
      scheduler.BroadcastExecute(TaskType::kSampling, batch_id);
    } else if (scheduled_batch->status == BatchStatus::kSampledStatus) {
      scheduled_batch->status = BatchStatus::kComputingStatus;

      scheduler.BroadcastExecute(TaskType::kPrepareInput, batch_id);
      if (using_precomputed_aggs_) { 
        scheduler.BroadcastExecute(TaskType::kPrepareAggregations, batch_id);
      }
    } else if (scheduled_batch->status == BatchStatus::kComputingStatus && is_first_batch) { // Only the first batch can proceed computation
      if (scheduled_batch->input_prepared && !scheduled_batch->input_computing) {
        scheduled_batch->input_computing = true;
        scheduler.BroadcastExecute(TaskType::kCompute, batch_id);
      }

      if (using_precomputed_aggs_ && scheduled_batch->aggregation_prepared && !scheduled_batch->aggregation_recomputing) {
        scheduled_batch->aggregation_recomputing = true;
        scheduler.BroadcastExecute(TaskType::kRecomputeAggregations, batch_id);
      }
    } else if (scheduled_batch->status == BatchStatus::kFirstLayerComputedStatus) { // Only for precomputed aggregations
      scheduled_batch->status = BatchStatus::kComputeRemainingStatus;
      scheduler.BroadcastExecute(TaskType::kComputeRemaining, batch_id);
    } else if (scheduled_batch->status == BatchStatus::kComputedStatus) {
      scheduled_batch->status = BatchStatus::kResultFetchingStatus;
      scheduler.BroadcastFetchResult(batch_id);
    } else if (scheduled_batch->status == BatchStatus::kFinishedStatus) {
      batch_finished = true;
    }

    if (batch_finished) {
      scheduled_batches_.erase(it++);
    } else {
      ++it;
      is_first_batch = false;
    }
  }
}

void VertexCutSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  auto it = scheduled_batches_.find(batch_id);
  auto& scheduled_batch = it->second;

  if (task_type == TaskType::kSampling) {
    scheduled_batch->status = kSampledStatus;
    TryScheduling(scheduler);
    return;
  }

  if (task_type == TaskType::kPrepareInput) {
    scheduled_batch->input_prepared = true;
  } else if (task_type == TaskType::kCompute) {
    assert(scheduled_batch->input_prepared && scheduled_batch->input_computing);
    scheduled_batch->input_computed = true;
  } else if (task_type == TaskType::kPrepareAggregations) {
    scheduled_batch->aggregation_prepared = true;
  } else if (task_type == TaskType::kRecomputeAggregations) {
    assert(scheduled_batch->aggregation_prepared && scheduled_batch->aggregation_recomputing);
    scheduled_batch->aggregation_recomputed = true;
  } else if (task_type == TaskType::kComputeRemaining) {
    assert(using_precomputed_aggs_);
    scheduled_batch->status = BatchStatus::kComputedStatus;
  }

  if (using_precomputed_aggs_) {
    if (scheduled_batch->input_computed && scheduled_batch->aggregation_recomputed &&
        scheduled_batch->status == BatchStatus::kComputingStatus) {
      scheduled_batch->status = BatchStatus::kFirstLayerComputedStatus;
    }
  } else {
    if (scheduled_batch->input_computed) {
      scheduled_batch->status = BatchStatus::kComputedStatus;
    }
  }

  TryScheduling(scheduler);
}

void VertexCutSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  auto it = scheduled_batches_.find(batch_id);
  assert(it != scheduled_batches_.end());
  assert(it->second->status == kInitializingStatus);
  it->second->status = kInitializedStatus;

  TryScheduling(scheduler);
}

void VertexCutSchedulingPolicy::OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) {
  auto req_id = batch_id_to_req_id_[batch_id];
  scheduler.ReportResult(req_id, result);

  auto it = scheduled_batches_.find(batch_id);
  assert(it != scheduled_batches_.end());
  assert(it->second->status == kResultFetchingStatus);
  it->second->status = kFinishedStatus;

  batch_id_to_req_id_.erase(batch_id);

  TryScheduling(scheduler);
}

}
}
