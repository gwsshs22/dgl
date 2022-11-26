#include "worker_process.h"

#include <iostream>
#include <thread>

#include "../execution/executor_actor.h"
#include "../execution/mpi/mpi_actor.h"
#include "../execution/mpi/gloo_rendezvous_actor.h"

#include "init_monitor_actor.h"
#include "process_control_actor.h"

namespace dgl {
namespace inference {

void WorkerProcessMain(caf::actor_system& system, const config& cfg) {
  auto master_host = GetEnv<std::string>(DGL_INFER_MASTER_HOST, "localhost");
  auto master_port = GetEnv<u_int16_t>(DGL_INFER_MASTER_PORT, 0);
  auto node_rank = GetEnv<int>(DGL_INFER_NODE_RANK, -1);
  auto num_nodes = GetEnv<int>(DGL_INFER_NUM_NODES, -1);
  auto num_backup_servers = GetEnv<int>(DGL_INFER_NUM_BACKUP_SERVERS, -1);
  auto num_devices_per_node = GetEnv<int>(DGL_INFER_NUM_DEVICES_PER_NODE, -1);
  auto iface = GetEnv<std::string>(DGL_INFER_IFACE, "");

  auto parallel_type = GetEnumEnv<ParallelizationType>(DGL_INFER_PARALLELIZATION_TYPE);
  auto using_precomputed_aggregations = GetEnv<bool>(DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS, false);
  auto result_dir = GetEnv<std::string>(DGL_INFER_RESULT_DIR, "");
  auto collect_stats = GetEnv<bool>(DGL_INFER_COLLECT_STATS, false);
  // TODO: env variables validation

  if (collect_stats) {
    EnableTracing();
  }

  auto node = retry<caf::node_id>([&] {
    return system.middleman().connect(master_host, master_port);
  });

  if (!node) {
    // TODO: error handling
    std::cerr << "*** connect failed: " << caf::to_string(node.error()) << std::endl;
    return;
  }

  auto global_init_mon_ptr = system.middleman().remote_lookup(caf::init_mon_atom_v, node.value());
  auto local_init_mon_proxy = system.spawn<init_monitor_proxy_actor>(global_init_mon_ptr);
  system.registry().put(caf::init_mon_atom_v, local_init_mon_proxy);

  auto fin_mon_actor_ptr = system.middleman().remote_lookup(caf::fin_mon_atom_v, node.value());
  auto fin_mon_actor = caf::actor_cast<caf::actor>(fin_mon_actor_ptr);

  auto process_mon_actor_ptr = system.middleman().remote_lookup(caf::process_mon_atom_v, node.value());
  system.registry().put(caf::process_mon_atom_v, process_mon_actor_ptr);

  auto process_creator = system.spawn(process_creator_fn);
  system.registry().put(caf::process_creator_atom_v, process_creator);

  auto gloo_ra_ptr = system.middleman().remote_lookup(caf::gloo_ra_atom_v, node.value());
  auto mpi_a = system.spawn<mpi_actor>(gloo_ra_ptr, MpiConfig {
    .rank = node_rank,
    .num_nodes = num_nodes,
    .hostname = master_host,
    .iface = iface,
  });

  auto exec_ctl_actor_ptr = system.middleman().remote_lookup(caf::exec_control_atom_v, node.value());

  auto executor = spawn_executor_actor(
      system,
      parallel_type,
      exec_ctl_actor_ptr,
      caf::actor_cast<caf::strong_actor_ptr>(mpi_a),
      node_rank,
      num_nodes,
      num_backup_servers,
      num_devices_per_node,
      result_dir,
      collect_stats,
      using_precomputed_aggregations);

  caf::scoped_actor self { system };
  std::cout << "Wait for the experiment is finished" << std::endl;
  auto rh = self->request(fin_mon_actor, std::chrono::minutes(10), caf::wait_atom_v);
  receive_result<bool>(rh);

  system.middleman().stop();
  // TODO: proper shutdown
  exit(0);
}

}
}
