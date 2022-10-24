#pragma once

#include <vector>

#include <caf/all.hpp>
#include <caf/io/all.hpp>

#include "mpi_actor.h"

namespace dgl {
namespace inference {

class mpi_control_actor : public caf::event_based_actor {

 public:
  mpi_control_actor(caf::actor_config& config, const int world_size);

 protected:
  caf::behavior make_behavior() override;

 private:
  caf::behavior make_running_behavior();

  caf::behavior running_;
  int world_size_;
  std::vector<MpiInitMsg> init_msgs_;
  std::vector<caf::actor> mpi_actors_;
};

}  // namespace inference
}  // namespace dgl
