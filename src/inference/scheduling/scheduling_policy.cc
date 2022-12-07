#include "scheduling_policy.h"

#include <cstdlib>
#include <limits>

namespace dgl {
namespace inference {

std::shared_ptr<SchedulingPolicy> CreatePolicy(ParallelizationType type,
                                               bool using_precomputed_aggs,
                                               int num_nodes,
                                               int num_devices_per_node,
                                               int num_samplers_per_node,
                                               bool execute_one_by_one) {
  if (type == ParallelizationType::kData) {
    return std::make_shared<DataSchedulingPolicy>(num_nodes, num_devices_per_node, num_samplers_per_node, execute_one_by_one);
  } else if (type == ParallelizationType::kP3) {
    return std::make_shared<P3SchedulingPolicy>(num_nodes, num_devices_per_node, num_samplers_per_node, execute_one_by_one);
  } else {
    return std::make_shared<VertexCutSchedulingPolicy>(num_nodes, num_devices_per_node, num_samplers_per_node, execute_one_by_one, using_precomputed_aggs);
  }
}

BaseSchedulingPolicy::BaseSchedulingPolicy(int num_nodes,
                                           int num_devices_per_node,
                                           int num_samplers_per_node,
                                           bool execute_one_by_one)
    : num_nodes_(num_nodes),
      num_devices_per_node_(num_devices_per_node),
      num_samplers_per_node_(num_samplers_per_node),
      execute_one_by_one_(execute_one_by_one) {
}

void BaseSchedulingPolicy::OnNewBatch(Scheduler& scheduler, 
                                      BatchInput&& input) {
  stats_.emplace(std::make_pair(input.req_id, RequestStats()));
  input_queue_.push(std::move(input));
  TryScheduling(scheduler);
}

int BaseSchedulingPolicy::IssueBatchId() {
  return batch_id_counter_++;
}

void BaseSchedulingPolicy::SetTaskDone(int req_id, TaskType task_type) {
}
  
void BaseSchedulingPolicy::ReportRequestDone(Scheduler& scheduler, int req_id, const NDArray& result) {
  auto it = stats_.find(req_id);
  assert(it != stats_.end());
  scheduler.ReportResult(req_id, result, it->second);
  stats_.erase(it);
}

//////////////////////////
// DataSchedulingPolicy //
//////////////////////////
DataSchedulingPolicy::DataSchedulingPolicy(int num_nodes,
                                           int num_devices_per_node,
                                           int num_samplers_per_node,
                                           bool execute_one_by_one)
   : BaseSchedulingPolicy(num_nodes, num_devices_per_node, num_samplers_per_node, execute_one_by_one) {

  for (int i = 0; i < num_nodes; i++) {
    scheduled_batches_.emplace_back(std::map<int, std::shared_ptr<ScheduledBatch>>());
    machines_.push_back(MachineStatus(num_devices_per_node));
  }

  if (execute_one_by_one) {
    max_concurrent_batches_ = 1;
  } else {
    max_concurrent_batches_ = num_samplers_per_node;
  }
}

void DataSchedulingPolicy::TryScheduling(Scheduler& scheduler) {
  while (!input_queue_.empty()) {
    int min_allocated_num = std::numeric_limits<int>::max();
    int node_rank = -1;

    if (execute_one_by_one_) {
      if (machines_[1].num_initializing > max_concurrent_batches_) {
        break;
      }

      node_rank = 1;
    } else {
      for (int i = 0; i < num_nodes_; i++) {
        if (machines_[i].num_initializing > max_concurrent_batches_) {
          continue;
        }
        if (machines_[i].num_allocated_batches < min_allocated_num) {
          min_allocated_num = machines_[i].num_allocated_batches;
          node_rank = i;
        }
      }

      if (node_rank == -1) {
        break;
      }
    }

    auto& min_allocated_machine = machines_[node_rank];

    min_allocated_machine.num_allocated_batches++;
    min_allocated_machine.num_initializing++;

    int batch_id = IssueBatchId();
    auto front = input_queue_.front();
    batch_id_to_req_id_[batch_id] = front.req_id;
    scheduler.LocalInitialize(batch_id, node_rank, front);
    input_queue_.pop();
 
    scheduled_batches_[node_rank].emplace(std::make_pair(batch_id, std::make_shared<ScheduledBatch>(batch_id)));
    batch_id_to_node_rank_[batch_id] = node_rank;
  }

  for (int node_rank = 0; node_rank < num_nodes_; node_rank++) {
    
    auto& scheduled_batches = scheduled_batches_[node_rank];
    auto& machine = machines_[node_rank];

    for (auto it = scheduled_batches.begin(); it != scheduled_batches.end();) {
      int batch_id = it->first;
      auto& scheduled_batch = it->second;
      auto batch_finished = false;

      if (scheduled_batch->status == BatchStatus::kInitializedStatus && machine.num_sampling < max_concurrent_batches_) {
        scheduled_batch->status = BatchStatus::kSamplingStatus;
        machine.num_sampling++;
        scheduler.LocalExecute(TaskType::kSampling, batch_id, node_rank, -1);
      } else if (scheduled_batch->status == BatchStatus::kSampledStatus && machine.num_computing < num_devices_per_node_) {
        scheduled_batch->status = BatchStatus::kComputingStatus;
        int gpu_idx = machine.GetIdleGpuIndex();
        machine.AssignGpu(gpu_idx);
        scheduled_batch->gpu_local_rank = gpu_idx;
        scheduler.LocalExecute(TaskType::kCompute, batch_id, node_rank, gpu_idx);
      } else if (scheduled_batch->status == BatchStatus::kComputedStatus && machine.num_fetching_result < max_concurrent_batches_) {
        scheduled_batch->status = BatchStatus::kFetchingResultStatus;
        machine.num_fetching_result++;
        scheduler.LocalFetchResult(batch_id, node_rank, -1);
      } else if (scheduled_batch->status == BatchStatus::kFinishedStatus) {
        batch_finished = true;
        scheduler.LocalExecute(TaskType::kCleanup, batch_id, node_rank, scheduled_batch->gpu_local_rank);
      }

      if (batch_finished) {
        scheduled_batches.erase(it++);
      } else {
        ++it;
      }
    }
  }
}

void DataSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  if (task_type == TaskType::kCleanup) {
    return;
  }

  auto node_rank = batch_id_to_node_rank_[batch_id];
  auto& machine = machines_[node_rank];
  auto it = scheduled_batches_[node_rank].find(batch_id);
  auto& scheduled_batch = it->second;

  if (task_type == TaskType::kSampling) {
    machine.num_sampling--;
    scheduled_batch->status = BatchStatus::kSampledStatus;
  } else if (task_type == TaskType::kCompute) {
    machine.FreeGpu(scheduled_batch->gpu_local_rank);
    scheduled_batch->status = BatchStatus::kComputedStatus;
  }

  TryScheduling(scheduler);
}

void DataSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  auto node_rank = batch_id_to_node_rank_[batch_id];
  auto& machine = machines_[node_rank];
  auto it = scheduled_batches_[node_rank].find(batch_id);
  assert(it != scheduled_batches_[node_rank].end());
  assert(it->second->status == kInitializingStatus);
  it->second->status = kInitializedStatus;

  machine.num_initializing--;
  TryScheduling(scheduler);
}

void DataSchedulingPolicy::OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) {
  auto req_id = batch_id_to_req_id_[batch_id];
  ReportRequestDone(scheduler, req_id, result);

  auto node_rank = batch_id_to_node_rank_[batch_id];
  auto& machine = machines_[node_rank];
  auto it = scheduled_batches_[node_rank].find(batch_id);
  assert(it != scheduled_batches_[node_rank].end());
  assert(it->second->status == kFetchingResultStatus);
  it->second->status = kFinishedStatus;

  machine.num_fetching_result--;
  machine.num_allocated_batches--;

  batch_id_to_node_rank_.erase(batch_id);
  batch_id_to_req_id_.erase(batch_id);

  TryScheduling(scheduler);
}

////////////////////////
// P3SchedulingPolicy //
////////////////////////
P3SchedulingPolicy::P3SchedulingPolicy(int num_nodes,
                                       int num_devices_per_node,
                                       int num_samplers_per_node,
                                       bool execute_one_by_one)
    : BaseSchedulingPolicy(num_nodes, num_devices_per_node, num_samplers_per_node, execute_one_by_one) {
  for (int i = 0; i < num_nodes; i++) {
    scheduled_batches_.emplace_back(std::map<int, std::shared_ptr<ScheduledBatch>>());
    machines_.push_back(MachineStatus(num_devices_per_node));
  }

  if (execute_one_by_one) {
    max_concurrent_batches_ = 1;
  } else {
    max_concurrent_batches_ = num_samplers_per_node;
  }

  latest_compute_done_node_rank_ = 0;
  compute_running_ = false;
}

void P3SchedulingPolicy::TryScheduling(Scheduler& scheduler) { 
  while (!input_queue_.empty()) {
    int min_allocated_num = std::numeric_limits<int>::max();
    int node_rank = -1;

    if (execute_one_by_one_) {
      if (machines_[1].num_initializing > max_concurrent_batches_) {
        break;
      }

      node_rank = 1;
    } else {
      for (int i = 0; i < num_nodes_; i++) {
        if (machines_[i].num_initializing > max_concurrent_batches_) {
          continue;
        }

        if (machines_[i].num_allocated_batches < min_allocated_num) {
          min_allocated_num = machines_[i].num_allocated_batches;
          node_rank = i;
        }
      }

      if (node_rank == -1) {
        break;
      }
    }

    auto& min_allocated_machine = machines_[node_rank];

    min_allocated_machine.num_allocated_batches++;
    min_allocated_machine.num_initializing++;

    int batch_id = IssueBatchId();
    auto front = input_queue_.front();
    batch_id_to_req_id_[batch_id] = front.req_id;
    scheduler.BroadcastInitialize(batch_id, BroadcastInitType::kScatterFeatureOnly, front);
    input_queue_.pop();

    scheduled_batches_[node_rank].emplace(std::make_pair(batch_id, std::make_shared<ScheduledBatch>(batch_id)));
    batch_id_to_node_rank_[batch_id] = node_rank;
  }

  int node_rank = rand() % num_nodes_;
  for (int i = 0; i < num_nodes_; i++) {
    node_rank++;
    node_rank %= num_nodes_;
    auto& scheduled_batches = scheduled_batches_[node_rank];
    auto& machine = machines_[node_rank];

    for (auto it = scheduled_batches.begin(); it != scheduled_batches.end();) {
      int batch_id = it->first;
      auto& scheduled_batch = it->second;
      auto batch_finished = false;

      if (scheduled_batch->status == BatchStatus::kInitializedStatus && machine.num_sampling < max_concurrent_batches_) {
        scheduled_batch->status = BatchStatus::kSamplingStatus;
        machine.num_sampling++;
        scheduler.LocalExecute(TaskType::kSampling, batch_id, node_rank, -1);
      } else if (scheduled_batch->status == BatchStatus::kSampledStatus && machine.num_push_computation_graph < max_concurrent_batches_) {
        scheduled_batch->status = BatchStatus::kPushingComputationGraphStatus;
        machine.num_push_computation_graph++;
        scheduler.BroadcastExecute(TaskType::kPushComputationGraph, batch_id, /* param0 = owner_node_rank */ node_rank, -1);
      } else if (scheduled_batch->status == BatchStatus::kPushedComputationGraphStatus && !compute_running_) {
        compute_running_ = true;
        latest_compute_done_node_rank_ = node_rank;
        scheduled_batch->status = BatchStatus::kComputingStatus;
        scheduler.BroadcastExecute(TaskType::kCompute, batch_id, /* param0 = owner_node_rank */ node_rank, /* param1 = owner_local_rank */ 0);
      } else if (scheduled_batch->status == BatchStatus::kComputedStatus && machine.num_fetching_result < max_concurrent_batches_) {
        machine.num_fetching_result++;
        scheduled_batch->status = BatchStatus::kFetchingResultStatus;
        scheduler.LocalFetchResult(batch_id, node_rank, -1);
      } else if (scheduled_batch->status == BatchStatus::kFinishedStatus) {
        batch_finished = true;
        scheduler.BroadcastExecute(TaskType::kCleanup, batch_id, node_rank, -1);
      }

      if (batch_finished) {
        scheduled_batches.erase(it++);
      } else {
        ++it;
      }
    }
  }
}

void P3SchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  if (task_type == TaskType::kCleanup) {
    return;
  }

  auto node_rank = batch_id_to_node_rank_[batch_id];
  auto& machine = machines_[node_rank];
  auto it = scheduled_batches_[node_rank].find(batch_id);
  auto& scheduled_batch = it->second;

  if (task_type == TaskType::kSampling) {
    machine.num_sampling--;
    scheduled_batch->status = BatchStatus::kSampledStatus;
  } else if (task_type == TaskType::kPushComputationGraph) {
    machine.num_push_computation_graph--;
    scheduled_batch->status = kPushedComputationGraphStatus;
  } else if (task_type == TaskType::kCompute) {
    compute_running_ = false;
    scheduled_batch->status = BatchStatus::kComputedStatus;
  }

  TryScheduling(scheduler);
}

void P3SchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  auto node_rank = batch_id_to_node_rank_[batch_id];
  auto& machine = machines_[node_rank];
  auto it = scheduled_batches_[node_rank].find(batch_id);
  assert(it != scheduled_batches_[node_rank].end());
  assert(it->second->status == kInitializingStatus);
  it->second->status = kInitializedStatus;

  machine.num_initializing--;
  TryScheduling(scheduler);
}

void P3SchedulingPolicy::OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) {
  auto req_id = batch_id_to_req_id_[batch_id];
  ReportRequestDone(scheduler, req_id, result);

  auto node_rank = batch_id_to_node_rank_[batch_id];
  auto& machine = machines_[node_rank];
  auto it = scheduled_batches_[node_rank].find(batch_id);
  assert(it != scheduled_batches_[node_rank].end());
  assert(it->second->status == kFetchingResultStatus);
  it->second->status = kFinishedStatus;

  machine.num_fetching_result--;
  machine.num_allocated_batches--;

  batch_id_to_node_rank_.erase(batch_id);
  batch_id_to_req_id_.erase(batch_id);

  TryScheduling(scheduler);
}

///////////////////////////////
// VertexCutSchedulingPolicy //
///////////////////////////////
VertexCutSchedulingPolicy::VertexCutSchedulingPolicy(int num_nodes,
                                                     int num_devices_per_node,
                                                     int num_samplers_per_node,
                                                     bool execute_one_by_one,
                                                     bool using_precomputed_aggs)
      : BaseSchedulingPolicy(num_nodes, num_devices_per_node, num_samplers_per_node, execute_one_by_one),
        using_precomputed_aggs_(using_precomputed_aggs) {
  if (execute_one_by_one) {
    max_concurrent_batches_ = 1;
    max_sampling_ = 1;
  } else {
    max_concurrent_batches_ = num_samplers_per_node;
    max_sampling_ = num_samplers_per_node;
  }

  for (int i = 0; i < max_sampling_; i++) {
    sampler_running_.push_back(false);
  }

  compute_running_ = false;
}

void VertexCutSchedulingPolicy::TryScheduling(Scheduler& scheduler) {
  while (!input_queue_.empty()) {
    if (machine_.num_initializing > max_concurrent_batches_) {
      break;
    }

    int batch_id = IssueBatchId();
    auto front = input_queue_.front();
    batch_id_to_req_id_[batch_id] = front.req_id;

    if (using_precomputed_aggs_) {
      scheduler.BroadcastInitialize(batch_id, BroadcastInitType::kAll, front);
    } else {
      scheduler.BroadcastInitialize(batch_id, BroadcastInitType::kScatter, front);
    }

    machine_.num_allocated_batches++;
    machine_.num_initializing++;
    input_queue_.pop();
 
    scheduled_batches_.emplace(std::make_pair(batch_id, std::make_shared<ScheduledBatch>(batch_id)));
  }

  for (auto it = scheduled_batches_.begin(); it != scheduled_batches_.end();) {
    int batch_id = it->first;
    auto& scheduled_batch = it->second;
    auto batch_finished = false;

    if (scheduled_batch->status == BatchStatus::kInitializedStatus && machine_.num_sampling < max_sampling_) {
      machine_.num_sampling++;
      scheduled_batch->status = BatchStatus::kSamplingStatus;

      int sampler_rank = -1;
      for (int i = 0; i < sampler_running_.size(); i++) {
        if (!sampler_running_[i]) {
          sampler_rank = i;
          break;
        }
      }

      CHECK(sampler_rank != -1);
      sampler_running_[sampler_rank] = true;
      scheduled_batch->sampler_rank = sampler_rank;
      scheduler.BroadcastExecute(TaskType::kSampling, batch_id, sampler_rank);
    } else if (scheduled_batch->status == BatchStatus::kSampledStatus && !compute_running_) {
      scheduled_batch->status = BatchStatus::kComputingStatus;
      compute_running_ = true;
      scheduler.BroadcastExecute(TaskType::kCompute, batch_id);
    } else if (scheduled_batch->status == BatchStatus::kComputedStatus && machine_.num_fetching_result < max_concurrent_batches_) {
      machine_.num_fetching_result++;
      scheduled_batch->status = BatchStatus::kFetchingResultStatus;
      scheduler.BroadcastFetchResult(batch_id);
    } else if (scheduled_batch->status == BatchStatus::kFinishedStatus) {
      batch_finished = true;
      scheduler.BroadcastExecute(TaskType::kCleanup, batch_id, scheduled_batch->sampler_rank);
    }

    if (batch_finished) {
      scheduled_batches_.erase(it++);
    } else {
      ++it;
    }
  }
}

void VertexCutSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  if (task_type == TaskType::kCleanup) {
    return;
  }

  auto it = scheduled_batches_.find(batch_id);
  auto& scheduled_batch = it->second;

  if (task_type == TaskType::kSampling) {
    scheduled_batch->status = kSampledStatus;
    sampler_running_[scheduled_batch->sampler_rank] = false;
    machine_.num_sampling--;
  } else if (task_type == TaskType::kCompute) {
    scheduled_batch->status = BatchStatus::kComputedStatus;
    compute_running_ = false;
  }

  TryScheduling(scheduler);
}

void VertexCutSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  auto it = scheduled_batches_.find(batch_id);
  assert(it != scheduled_batches_.end());
  assert(it->second->status == kInitializingStatus);
  it->second->status = kInitializedStatus;

  machine_.num_initializing--;
  TryScheduling(scheduler);
}

void VertexCutSchedulingPolicy::OnFinished(Scheduler& scheduler, int batch_id, const NDArray& result) {
  auto req_id = batch_id_to_req_id_[batch_id];
  ReportRequestDone(scheduler, req_id, result);

  auto it = scheduled_batches_.find(batch_id);
  assert(it != scheduled_batches_.end());
  assert(it->second->status == kFetchingResultStatus);
  it->second->status = kFinishedStatus;

  machine_.num_fetching_result--;
  machine_.num_allocated_batches--;

  batch_id_to_req_id_.erase(batch_id);

  TryScheduling(scheduler);
}

}
}
