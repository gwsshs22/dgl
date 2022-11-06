#include "scheduling_policy.h"

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

// DataSchedulingPolicy
DataSchedulingPolicy::DataSchedulingPolicy(bool using_precomputed_aggs,
                                           int num_nodes,
                                           int num_devices_per_node)
   : BaseSchedulingPolicy(using_precomputed_aggs, num_nodes, num_devices_per_node) {

}

void DataSchedulingPolicy::TryScheduling(Scheduler& scheduler) {
}

void DataSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  TryScheduling(scheduler);
}

void DataSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {

}

// P3SchedulingPolicy
void P3SchedulingPolicy::TryScheduling(Scheduler& scheduler) { 
  
}

void P3SchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
  TryScheduling(scheduler);
}

void P3SchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  
}

// VertexCutSchedulingPolicy
void VertexCutSchedulingPolicy::TryScheduling(Scheduler& scheduler) {
  while (!input_queue_.empty()) {
    int batch_id = IssueBatchId();
    scheduler.BroadcastInitialize(batch_id, input_queue_.front());
    input_queue_.pop();
  }
  // for (auto const& p : scheduled_batches_) {
  //   auto& scheduled_batch = p.second;
  //   if (scheduled_batch->status() == ScheduledBatch::Status::kCreated) {
  //     scheduled_batch->SetStatus(ScheduledBatch::Status::kInitializing);
  //     const auto& batch_input = scheduled_batch->batch_input();
  //     send(exec_ctl_actor_, caf::init_atom_v, scheduled_batch->batch_id(),
  //         batch_input.new_gnids, batch_input.src_gnids, batch_input.dst_gnids);
  //   }
  // }
}

void VertexCutSchedulingPolicy::OnExecuted(Scheduler& scheduler, int batch_id, TaskType task_type) {
      // if (task_type == TaskType::kInitialize) {
      //   scheduled_batches_[batch_id]->SetStatus(ScheduledBatch::Status::kReady);
      //   send(exec_ctl_actor_, caf::exec_atom_v, TaskType::kInputBroadcast, batch_id);
      // } else if (task_type == TaskType::kInputBroadcast) {
      //   send(exec_ctl_actor_, caf::exec_atom_v, TaskType::kTest, batch_id);
      // }
      // TrySchedule();
  std::cerr << "batch_id=" << batch_id << " task_type=" << task_type << " done." << std::endl;
  TryScheduling(scheduler);
}

void VertexCutSchedulingPolicy::OnInitialized(Scheduler& scheduler, int batch_id) {
  scheduler.BroadcastExecute(TaskType::kTest, batch_id);
}

}
}
