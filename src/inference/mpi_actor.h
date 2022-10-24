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
class mpi_actor : public caf::event_based_actor {

 public:
  mpi_actor(caf::actor_config& config,
            const caf::strong_actor_ptr& control_actor_ptr,
            const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
            const MpiConfig& mpi_config);

  static caf::actor spawn(caf::actor_system& system,
                          const caf::strong_actor_ptr& control_actor_ptr,
                          const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
                          const MpiConfig& mpi_config) {
    auto new_actor = system.spawn<mpi_actor, caf::detached>(control_actor_ptr, gloo_rendezvous_actor_ptr, mpi_config);
    caf::anon_send(new_actor, caf::init_atom_v);
    return new_actor;
  }

 protected:
  caf::behavior make_behavior() override;
 private:
  // MpiInitMsg Initialize() override;

  // void OnMessage(const int& msg) override;
  caf::behavior make_initializing();
  caf::behavior make_running();

  caf::behavior initializing_;
  caf::behavior running_;

  const caf::strong_actor_ptr control_actor_ptr_;
  const caf::strong_actor_ptr gloo_rendezvous_actor_ptr_;
  const MpiConfig mpi_config_;

  std::shared_ptr<gloo::transport::Device> gloo_device_;
  std::shared_ptr<gloo::rendezvous::Context> gloo_context_;

  int rank_ = -1;
  int world_size_ = -1;
  
};



}  // namespace inference
}  // namespace dgl
