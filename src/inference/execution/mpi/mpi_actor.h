#pragma once

#include <iostream>
#include <thread>

#include <caf/all.hpp>
#include <caf/io/all.hpp>
#include <gloo/rendezvous/context.h>
#include <gloo/transport/tcp/device.h>

#include <dgl/inference/common.h>

#include "gloo_rendezvous_actor.h"
#include "gloo_executor.h"

namespace dgl {
namespace inference {

struct MpiConfig {
  int rank = -1;
  int num_nodes = -1;
  std::string hostname = "";
  std::string iface = "";
};

// gloo based mpi actor
class mpi_actor : public caf::event_based_actor {

 public:
  mpi_actor(caf::actor_config& config,
            const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
            const MpiConfig& mpi_config);

 private:
  // void act() override;
  caf::behavior make_behavior();

  const caf::strong_actor_ptr gloo_rendezvous_actor_ptr_;
  std::shared_ptr<GlooExecutor> gloo_executor_;
  const MpiConfig mpi_config_;
};

}  // namespace inference
}  // namespace dgl
