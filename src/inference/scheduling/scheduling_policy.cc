#include "scheduling_policy.h"

#include <limits>

namespace dgl {
namespace inference {

std::shared_ptr<SchedulingPolicy> CreatePolicy(ParallelizationType type,
                                               bool using_precomputed_aggs,
                                               int num_nodes,
                                               int num_devices_per_node) {
  if (type == ParallelizationType::kData) {
    return std::make_shared<DataSchedulingPolicy>(using_precomputed_aggs, num_nodes, num_devices_per_node);
  } else if (type == ParallelizationType::kP3) {
    return std::make_shared<P3SchedulingPolicy>(using_precomputed_aggs, num_nodes, num_devices_per_node);
  } else {
    return std::make_shared<VertexCutSchedulingPolicy>(using_precomputed_aggs, num_nodes, num_devices_per_node);
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
    scheduler.LocalInitialize(batch_id, node_rank, input_queue_.front());
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
      bool batch_finished = false;

      if (scheduled_batch->status == BatchStatus::kInitialized) {
        scheduled_batch->status = BatchStatus::kSampling;
        scheduler.LocalExecute(TaskType::kSampling, batch_id, node_rank, local_rank);
      } else if (scheduled_batch->status == BatchStatus::kSampled) {
        scheduled_batch->status = BatchStatus::kComputing;

        scheduler.LocalExecute(TaskType::kPrepareInput, batch_id, node_rank, local_rank);
        if (using_precomputed_aggs_) { 
          scheduler.LocalExecute(TaskType::kPrepareAggregations, batch_id, node_rank, local_rank);
        }
      } else if (scheduled_batch->status == BatchStatus::kComputing && is_first_batch) { // Only the first batch can proceed computation
        if (scheduled_batch->input_prepared && !scheduled_batch->input_computing) {
          scheduled_batch->input_computing = true;
          scheduler.LocalExecute(TaskType::kCompute, batch_id, node_rank, local_rank);
        }

        if (using_precomputed_aggs_ && scheduled_batch->aggregation_prepared && !scheduled_batch->aggregation_recomputing) {
          scheduled_batch->aggregation_recomputing = true;
          scheduler.LocalExecute(TaskType::kRecomputeAggregations, batch_id, node_rank, local_rank);
        }
      } else if (scheduled_batch->status == BatchStatus::kFirstLayerComputed) { // Only for precomputed aggregations
        scheduled_batch->status = BatchStatus::kComputeRemaining;
        scheduler.LocalExecute(TaskType::kComputeRemaining, batch_id, node_rank, local_rank);
      } else if (scheduled_batch->status == BatchStatus::kComputed) {
        std::cerr << "[DONE] batch_id= " << batch_id << std::endl;
        batch_finished = true;
      }

      is_first_batch = false;

      if (batch_finished) {
        scheduled_batches.erase(it++);
      } else {
        ++it;
      }
    }
  }
}

void DataSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  auto global_rank = batch_id_to_global_rank_[batch_id];
  auto it = scheduled_batches_[global_rank].find(batch_id);
  auto& scheduled_batch = it->second;

  if (task_type == TaskType::kSampling) {
    scheduled_batch->status = kSampled;
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
    scheduled_batch->status = BatchStatus::kComputed;
  }

  if (using_precomputed_aggs_) {
    if (scheduled_batch->input_computed && scheduled_batch->aggregation_recomputed &&
        scheduled_batch->status == BatchStatus::kComputing) {
      scheduled_batch->status = BatchStatus::kFirstLayerComputed;
    }
  } else {
    if (scheduled_batch->input_computed) {
      scheduled_batch->status = BatchStatus::kComputed;
    }
  }

  TryScheduling(scheduler);
}

void DataSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  auto global_rank = batch_id_to_global_rank_[batch_id];
  auto it = scheduled_batches_[global_rank].find(batch_id);
  assert(it != scheduled_batches_[global_rank].end());
  assert(it->second->status == kInitializing);
  it->second->status = kInitialized;

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

///////////////////////////////
// VertexCutSchedulingPolicy //
///////////////////////////////
void VertexCutSchedulingPolicy::TryScheduling(Scheduler& scheduler) {
  while (!input_queue_.empty()) {
    int batch_id = IssueBatchId();
    scheduler.BroadcastInitialize(batch_id, input_queue_.front());
    input_queue_.pop();
  }
}

void VertexCutSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  TryScheduling(scheduler);
}

void VertexCutSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  scheduler.BroadcastExecute(TaskType::kTest, batch_id);
}

}
}
