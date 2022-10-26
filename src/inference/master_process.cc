#include "master_process.h"

#include <iostream>

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

void MasterProcessMain(caf::actor_system& system, const config& cfg) {
  auto res = system.middleman().open(cfg.port);
  if (!res) {
    std::cerr << "*** cannot open port: " << caf::to_string(res.error()) << std::endl;
    return;
  }

  auto init_mon_actor = system.spawn<init_monitor_actor>(2);
  system.registry().put(caf::init_mon_atom_v, init_mon_actor);

  auto gloo_ra = system.spawn<gloo_rendezvous_actor>();
  system.registry().put(caf::gloo_ra_atom_v, gloo_ra);
  auto gloo_ra_ptr = system.registry().get(caf::gloo_ra_atom_v);

  auto mpi_a = mpi_actor::spawn(system, gloo_ra_ptr, MpiConfig {
    .rank = 0,
    .world_size = 2,
    .hostname = "localhost",
    .iface = "eno4",
  });

  caf::scoped_actor self { system };
  auto required_actors = std::vector<std::string>();
  required_actors.emplace_back("mpi");
  self->request(init_mon_actor, std::chrono::seconds(30), caf::wait_atom_v, required_actors);

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}