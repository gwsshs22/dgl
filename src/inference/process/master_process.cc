#include "master_process.h"

#include <iostream>

#include "../scheduling/scheduler_actor.h"
#include "../execution/executor_control_actor.h"
#include "../execution/executor_actor.h"
#include "../execution/mpi/mpi_actor.h"
#include "../execution/mpi/gloo_rendezvous_actor.h"

#include "init_monitor_actor.h"
#include "process_control_actor.h"


#include "../execution/gnn/gnn_executor.h"

namespace dgl {
namespace inference {

void MasterProcessMain(caf::actor_system& system, const config& cfg) {
  auto master_host = GetEnv<std::string>(DGL_INFER_MASTER_HOST, "localhost");
  auto master_port = GetEnv<u_int16_t>(DGL_INFER_MASTER_PORT, 0);
  auto node_rank = GetEnv<int>(DGL_INFER_NODE_RANK, -1);
  auto num_nodes = GetEnv<int>(DGL_INFER_NUM_NODES, -1);
  auto num_devices_per_node = GetEnv<int>(DGL_INFER_NUM_DEVICES_PER_NODE, -1);
  auto iface = GetEnv<std::string>(DGL_INFER_IFACE, "");

  auto parallel_type = GetEnumEnv<ParallelizationType>(DGL_INFER_PARALLELIZATION_TYPE);
  auto using_precomputed_aggregations = GetEnv<bool>(DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS, false);
  // TODO: env variables validation

  auto init_mon_actor = system.spawn<init_monitor_actor>();
  system.registry().put(caf::init_mon_atom_v, init_mon_actor);

  auto process_mon_actor = system.spawn(process_monitor_fn);
  system.registry().put(caf::process_mon_atom_v, process_mon_actor);

  auto process_creator = system.spawn(process_creator_fn);
  system.registry().put(caf::process_creator_atom_v, process_creator);

  auto gloo_ra = system.spawn<gloo_rendezvous_actor>();
  system.registry().put(caf::gloo_ra_atom_v, gloo_ra);
  auto gloo_ra_ptr = system.registry().get(caf::gloo_ra_atom_v);

  auto node_port = retry<uint16_t>([&] {
    return system.middleman().open(master_port);
  });

  if (!node_port) {
    // TODO: error handling
    std::cerr << "*** cannot open port: " << caf::to_string(node_port.error()) << std::endl;
    exit(-1);
  }

  auto mpi_a = system.spawn<mpi_actor>(gloo_ra_ptr, MpiConfig {
    .rank = node_rank,
    .num_nodes = num_nodes,
    .hostname = master_host,
    .iface = iface,
  });

  auto mpi_actor_ptr = caf::actor_cast<caf::strong_actor_ptr>(mpi_a);

  auto exec_ctl_actor = system.spawn<executor_control_actor>(mpi_actor_ptr);
  system.registry().put(caf::exec_control_atom_v, exec_ctl_actor);
  auto exec_ctl_actor_ptr = system.registry().get(caf::exec_control_atom_v);

  auto executor = system.spawn<executor_actor>(
      exec_ctl_actor_ptr,
      mpi_actor_ptr,
      node_rank,
      num_nodes,
      num_devices_per_node);

  auto required_actors = std::vector<std::string>({ "mpi", "exec_ctrl" });
  auto scheduler = system.spawn<scheduler_actor>(exec_ctl_actor_ptr,
                                                 parallel_type,
                                                 using_precomputed_aggregations,
                                                 num_nodes,
                                                 num_devices_per_node);

  caf::scoped_actor self { system };
  auto res_hdl = self->request(init_mon_actor, std::chrono::seconds(30), caf::wait_atom_v, required_actors);
  receive_result<bool>(res_hdl);

  std::cerr << "All services initialized." << std::endl;

  auto cpu_context = DLContext { kDLCPU, 0 };
  NDArray new_gnids = NDArray::FromVector(std::vector<int32_t>{ 0, 1, 2, 3 }, cpu_context);
  NDArray src_gnids = NDArray::FromVector(std::vector<int32_t>{ 1, 1, 2, 0, 3 }, cpu_context);
  NDArray dst_gnids = NDArray::FromVector(std::vector<int32_t>{ 0, 1, 3, 3, 0 }, cpu_context);

  for (int j = 0; j < 4; j++) {
  for (int i = 0; i < 4; i++) {
    anon_send(scheduler, caf::enqueue_atom_v, new_gnids, src_gnids, dst_gnids);
  }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  // anon_send(scheduler, caf::enqueue_atom_v, new_gnids, src_gnids, dst_gnids);
  // anon_send(scheduler, caf::enqueue_atom_v, new_gnids, src_gnids, dst_gnids);

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}
