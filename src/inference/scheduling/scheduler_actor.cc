#include "scheduler_actor.h"

namespace dgl {
namespace inference {

scheduler_actor::scheduler_actor(caf::actor_config& config,
                                   const caf::strong_actor_ptr& exec_ctl_actor_ptr)
    : event_based_actor(config) {
  exec_ctl_actor_ = caf::actor_cast<caf::actor>(exec_ctl_actor_ptr);
}

caf::behavior scheduler_actor::make_behavior() {
  return {
    [&](caf::enqueue_atom,
        const NDArray& new_gnids,
        const NDArray& src_gnids,
        const NDArray& dst_gnids) {
      input_queue_.push(BatchInput { new_gnids, src_gnids, dst_gnids });
      TrySchedule();
    }
  };
}

void scheduler_actor::TrySchedule() {
  // TODO: Better batching
  while (!input_queue_.empty()) {
    int batch_id = batch_id_counter_++;
    scheduled_batches_.insert(std::make_pair(batch_id, std::make_unique<ScheduledBatch>(batch_id, input_queue_.front())));
    input_queue_.pop();
  }

  for (auto const& p : scheduled_batches_) {
    auto& scheduled_batch = p.second;
    if (scheduled_batch->status() == ScheduledBatch::Status::kCreated) {
      scheduled_batch->SetStatus(ScheduledBatch::Status::kInitializing);
      const auto& batch_input = scheduled_batch->batch_input();
      send(exec_ctl_actor_, caf::init_atom_v, scheduled_batch->batch_id(),
          batch_input.new_gnids, batch_input.src_gnids, batch_input.dst_gnids);
    }
  }

}

}
}
