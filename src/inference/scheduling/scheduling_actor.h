#pragma once

#include <dgl/inference/actor_types.h>

namespace dgl {
namespace inference {

class scheduling_actor : public caf::event_based_actor {

 public:
  scheduling_actor(caf::actor_config& config,
                   const caf::strong_actor_ptr& exec_ctl_actor_ptr);

 private:
  caf::behavior make_behavior() override;
  
  void TrySchedule();

  caf::strong_actor_ptr exec_ctl_actor_ptr_;
  int batch_id_counter = 0;
};

}
}
