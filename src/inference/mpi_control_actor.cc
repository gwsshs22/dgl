#include "mpi_control_actor.h"

#include <iostream>
#include <algorithm>

#include "actor/actor_types.h"

namespace dgl {
namespace inference {

mpi_control_actor::mpi_control_actor(caf::actor_config& config, const int world_size)
    : event_based_actor(config),
      world_size_(world_size) {
  running_.assign(make_running_behavior());
}

caf::behavior mpi_control_actor::make_behavior() {
  return {
    [=](caf::initialized_atom, const MpiInitMsg& init_msg) {
      init_msgs_.push_back(init_msg);
      if (init_msgs_.size() == world_size_) {
        std::sort(init_msgs_.begin(), init_msgs_.end(), [&](const MpiInitMsg& m1, const MpiInitMsg& m2) {
          return m1.rank < m2.rank;
        });

        int rank = 0;
        for (const auto& m : init_msgs_) {
          CAF_ASSERT(rank == m.rank);
          CAF_ASSERT(world_size_ == m.world_size);
          mpi_actors_.push_back(caf::actor_cast<caf::actor>(m.actor_ptr));
          rank++;
        }
        init_msgs_.clear();

        become(running_);
      }
    }
  };
}

caf::behavior mpi_control_actor::make_running_behavior() {
  return {
    [=](int) {
      for (const auto& actor : mpi_actors_) {
        send(actor, 10);
      }
    }
  };
}

}
}