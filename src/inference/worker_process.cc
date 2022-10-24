#include <iostream>
#include <thread>

#include <gloo/allreduce_ring.h>
#include <gloo/broadcast.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/transport/tcp/device.h>

#include "mpi_control_actor.h"
#include "mpi_actor.h"
#include "worker_process.h"
#include "gloo_rendezvous_actor.h"

namespace dgl {
namespace inference {

void WorkerProcessMain(caf::actor_system& system, const config& cfg) {
  auto node = system.middleman().connect(cfg.host, cfg.port);
  if (!node) {
    std::cerr << "*** connect failed: " << caf::to_string(node.error()) << std::endl;
    return;
  }

  auto mpi_control_actor_ptr = system.middleman().remote_lookup(caf::mpi_ctr_atom_v, node.value());
  auto gloo_ra_ptr = system.middleman().remote_lookup(caf::gloo_ra_atom_v, node.value());
  auto mpi_a = mpi_actor::spawn(system, mpi_control_actor_ptr, gloo_ra_ptr, MpiConfig {
    .hostname = "localhost",
    .iface = "eno4",
    .rank = 1,
    .world_size = 2
  });

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}