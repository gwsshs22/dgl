#include "executor_control_actor.h"

#include <algorithm>
#include <iostream>

namespace dgl {
namespace inference {

executor_control_actor::executor_control_actor(caf::actor_config& config)
    : event_based_actor(config) {
}

caf::behavior executor_control_actor::make_behavior() {
  return initializing();
}

caf::behavior executor_control_actor::initializing() {
  return {
    [=](caf::initialized_atom, const caf::strong_actor_ptr& executor_ptr, int executor_rank, int world_size) {
      assert(world_size >= 1);
      assert(executor_rank >= 0);
      assert(executor_rank < world_size);
      for (auto& p : executors_) {
        assert(p.second != executor_rank);
      }

      if (world_size_ < 0) {
        world_size_ = world_size;
      } else {
        assert(world_size == world_size_);
      }

      executors_.push_back(std::make_pair(executor_ptr, executor_rank));

      if (executors_.size() == world_size_) {
        std::sort(executors_.begin(), executors_.end(), [](const auto& e1, const auto& e2) { return e1.second < e2.second; });
        become(running());
      }
    }
  };
}

caf::behavior executor_control_actor::running() {
  ReportToInitMon(*this, "exec_ctrl", 0, 1);
  return {
    [&](caf::init_atom, int batch_id, const NDArray& new_gnids, const NDArray& src_gnids, const NDArray& dst_gnids) {
      std::cout << batch_id << " " << new_gnids << " " << src_gnids << " " << dst_gnids;
    }
  };
}

}
}
