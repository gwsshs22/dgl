#pragma once

#include <queue>
#include <map>
#include <dgl/inference/common.h>

#include "scheduled_batch.h"

namespace dgl {
namespace inference {

class scheduler_actor : public caf::event_based_actor {

 public:
  scheduler_actor(caf::actor_config& config,
                   const caf::strong_actor_ptr& exec_ctl_actor_ptr);

 private:
  caf::behavior make_behavior() override;
  
  void TrySchedule();
  caf::actor exec_ctl_actor_;
  int batch_id_counter_ = 0;

  std::queue<BatchInput> input_queue_;
  std::map<int, std::unique_ptr<ScheduledBatch>> scheduled_batches_;
};

}
}
