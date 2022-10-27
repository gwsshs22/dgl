#include "master_process.h"

#include <iostream>

#include <gloo/allreduce_ring.h>
#include <gloo/broadcast.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/transport/tcp/device.h>

#include "./scheduling/scheduler_actor.h"
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

  auto executor = system.spawn<executor_actor>(exec_ctl_actor_ptr, caf::actor_cast<caf::strong_actor_ptr>(mpi_a), rank, world_size);

  auto required_actors = std::vector<std::string>({ "mpi", "exec_ctrl" });
  auto scheduler = system.spawn<scheduler_actor>(exec_ctl_actor_ptr);

  auto res_hdl = self->request(init_mon_actor, std::chrono::seconds(30), caf::wait_atom_v, required_actors);
  receive_result<bool>(res_hdl);


  std::cerr << "All services initialized." << std::endl;

  auto cpu_context = DLContext { kDLCPU, 0 };
  NDArray new_gnids = NDArray::FromVector(std::vector<int32_t>{ 0, 1, 2, 3 }, cpu_context);
  NDArray src_gnids = NDArray::FromVector(std::vector<int32_t>{ 1, 1, 2, 0, 3 }, cpu_context);
  NDArray dst_gnids = NDArray::FromVector(std::vector<int32_t>{ 0, 1, 3, 3, 0 }, cpu_context);

  anon_send(scheduler, caf::enqueue_atom_v, new_gnids, src_gnids, dst_gnids);
  anon_send(scheduler, caf::enqueue_atom_v, new_gnids, src_gnids, dst_gnids);
  anon_send(scheduler, caf::enqueue_atom_v, new_gnids, src_gnids, dst_gnids);

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}