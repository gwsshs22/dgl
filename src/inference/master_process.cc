#include "master_process.h"

#include <iostream>

#include <gloo/allreduce_ring.h>
#include <gloo/broadcast.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/transport/tcp/device.h>

#include "./scheduling/scheduling_actor.h"
#include "./execution/executor_control_actor.h"
#include "./execution/executor_actor.h"
#include "./mpi/mpi_actor.h"
#include "./mpi/gloo_rendezvous_actor.h"

#include "init_monitor_actor.h"

namespace dgl {
namespace inference {

void MasterProcessMain(caf::actor_system& system, const config& cfg) {
  caf::scoped_actor self { system };

  auto init_mon_actor = system.spawn<init_monitor_actor>();
  system.registry().put(caf::init_mon_atom_v, init_mon_actor);

  auto exec_ctl_actor = system.spawn<executor_control_actor>();
  system.registry().put(caf::exec_control_atom_v, exec_ctl_actor);
  auto exec_ctl_actor_ptr = system.registry().get(caf::exec_control_atom_v);

  auto gloo_ra = system.spawn<gloo_rendezvous_actor>();
  system.registry().put(caf::gloo_ra_atom_v, gloo_ra);
  auto gloo_ra_ptr = system.registry().get(caf::gloo_ra_atom_v);

  auto res = system.middleman().open(cfg.port);
  if (!res) {
    std::cerr << "*** cannot open port: " << caf::to_string(res.error()) << std::endl;
    return;
  }

  int rank = 0;
  int world_size = 2;

  auto mpi_a = system.spawn<mpi_actor>(gloo_ra_ptr, MpiConfig {
    .rank = rank,
    .world_size = world_size,
    .hostname = "localhost",
    .iface = "eno4",
  });

  auto executor = system.spawn<executor_actor>(exec_ctl_actor_ptr, rank, world_size);

  auto required_actors = std::vector<std::string>({ "mpi", "exec_ctrl" });
  auto res_hdl = self->request(init_mon_actor, std::chrono::seconds(30), caf::wait_atom_v, required_actors);
  receive_result<bool>(res_hdl);

  auto scheduler = system.spawn<scheduling_actor>(exec_ctl_actor_ptr);

  std::cerr << "All services initialized. Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}