#include "master_process.h"

#include <iostream>

#include <gloo/allreduce_ring.h>
#include <gloo/broadcast.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/transport/tcp/device.h>

#include "gloo_rendezvous_actor.h"
#include "mpi_control_actor.h"
#include "mpi_actor.h"

namespace dgl {
namespace inference {

void MasterProcessMain(caf::actor_system& system, const config& cfg) {
  auto res = system.middleman().open(cfg.port);
  if (!res) {
    std::cerr << "*** cannot open port: " << caf::to_string(res.error()) << std::endl;
    return;
  }

  auto gloo_ra = system.spawn<gloo_rendezvous_actor>();
  system.registry().put(caf::gloo_ra_atom_v, gloo_ra);
  auto gloo_ra_ptr = system.registry().get(caf::gloo_ra_atom_v);


  auto mpi_control_a = system.spawn<mpi_control_actor>(2);
  system.registry().put(caf::mpi_ctr_atom_v, mpi_control_a);
  auto mpi_control_actor_ptr = system.registry().get(caf::mpi_ctr_atom_v);

  auto mpi_a = mpi_actor::spawn(system, mpi_control_actor_ptr, gloo_ra_ptr, MpiConfig {
    .hostname = "localhost",
    .iface = "eno4",
    .rank = 0,
    .world_size = 2
  });


  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  std::cerr << "now send mpi message" << std::endl;
  caf::anon_send(mpi_control_a, 5);

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}