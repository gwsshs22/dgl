#include "worker_process.h"

#include <iostream>
#include <thread>

#include <gloo/allreduce_ring.h>
#include <gloo/broadcast.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/transport/tcp/device.h>

#include "./mpi/mpi_actor.h"
#include "./mpi/gloo_rendezvous_actor.h"

#include "init_monitor_actor.h"

namespace dgl {
namespace inference {

void WorkerProcessMain(caf::actor_system& system, const config& cfg) {
  auto node = system.middleman().connect(cfg.host, cfg.port);
  if (!node) {
    std::cerr << "*** connect failed: " << caf::to_string(node.error()) << std::endl;
    return;
  }

  auto global_init_mon_ptr = system.middleman().remote_lookup(caf::init_mon_atom_v, node.value());
  auto local_init_mon_proxy = system.spawn<init_monitor_proxy_actor>(global_init_mon_ptr);
  system.registry().put(caf::init_mon_atom_v, local_init_mon_proxy);

  auto gloo_ra_ptr = system.middleman().remote_lookup(caf::gloo_ra_atom_v, node.value());
  auto mpi_a = mpi_actor::spawn(system, gloo_ra_ptr, MpiConfig {
    .rank = 1,
    .world_size = 2,
    .hostname = "localhost",
    .iface = "eno4",
  });

  caf::scoped_actor self { system };
  auto required_actors = std::vector<std::string>();
  required_actors.emplace_back("mpi");
  self->request(local_init_mon_proxy, std::chrono::seconds(30), caf::wait_atom_v, required_actors);

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}