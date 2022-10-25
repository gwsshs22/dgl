#pragma once

#include <iostream>
#include <thread>

#include <caf/all.hpp>
#include <caf/io/all.hpp>
#include <gloo/rendezvous/context.h>
#include <gloo/transport/tcp/device.h>

#include "./actor/actor_types.h"
#include "./actor/custom_types.h"

namespace dgl {
namespace inference {

struct MpiConfig {
  std::string hostname = "";
  std::string iface = "";
  int rank = -1;
  int world_size = -1;
};

// gloo based mpi actor
class mpi_actor : public caf::blocking_actor {

 public:
  mpi_actor(caf::actor_config& config,
            const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
            const MpiConfig& mpi_config);

  static caf::actor spawn(caf::actor_system& system,
                          const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
                          const MpiConfig& mpi_config) {
    auto new_actor = system.spawn<mpi_actor, caf::detached>(gloo_rendezvous_actor_ptr, mpi_config);
    return new_actor;
  }

 private:
  void act() override;

  const caf::strong_actor_ptr gloo_rendezvous_actor_ptr_;
  const MpiConfig mpi_config_;
};



}  // namespace inference
}  // namespace dgl
