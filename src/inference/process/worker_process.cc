#include "worker_process.h"

#include <iostream>
#include <thread>

#include <gloo/allreduce_ring.h>
#include <gloo/broadcast.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/transport/tcp/device.h>

#include "../execution/executor_actor.h"
#include "../execution/mpi/mpi_actor.h"
#include "../execution/mpi/gloo_rendezvous_actor.h"

#include "../init_monitor_actor.h"

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

  auto exec_ctl_actor_ptr = system.middleman().remote_lookup(caf::exec_control_atom_v, node.value());

  int rank = 1;
  int world_size = 2;

  auto gloo_ra_ptr = system.middleman().remote_lookup(caf::gloo_ra_atom_v, node.value());
  auto mpi_a = system.spawn<mpi_actor>(gloo_ra_ptr, MpiConfig {
    .rank = rank,
    .world_size = world_size,
    .hostname = "localhost",
    .iface = "eno4",
  });

  auto executor = system.spawn<executor_actor>(exec_ctl_actor_ptr, caf::actor_cast<caf::strong_actor_ptr>(mpi_a), rank, world_size);

  // caf::scoped_actor self { system };
  // auto required_actors = std::vector<std::string>();
  // required_actors.emplace_back("mpi");
  // self->request(local_init_mon_proxy, std::chrono::seconds(30), caf::wait_atom_v, required_actors);

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}