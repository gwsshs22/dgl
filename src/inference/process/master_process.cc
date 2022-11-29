#include "master_process.h"

#include <iostream>

#include "../scheduling/scheduler_actor.h"
#include "../execution/executor_control_actor.h"
#include "../execution/executor_actor.h"
#include "../execution/mpi/mpi_actor.h"
#include "../execution/mpi/gloo_rendezvous_actor.h"
#include "../execution/gnn/gnn_executor.h"

#include "init_monitor_actor.h"
#include "process_control_actor.h"
#include "experiments.h"

namespace dgl {
namespace inference {

void MasterProcessMain(caf::actor_system& system, const config& cfg) {
  auto master_host = GetEnv<std::string>(DGL_INFER_MASTER_HOST, "localhost");
  auto master_port = GetEnv<u_int16_t>(DGL_INFER_MASTER_PORT, 0);
  auto node_rank = GetEnv<int>(DGL_INFER_NODE_RANK, -1);
  auto num_nodes = GetEnv<int>(DGL_INFER_NUM_NODES, -1);
  auto num_backup_servers = GetEnv<int>(DGL_INFER_NUM_BACKUP_SERVERS, -1);
  auto num_devices_per_node = GetEnv<int>(DGL_INFER_NUM_DEVICES_PER_NODE, -1);
  auto num_samplers_per_node = GetEnv<int>(DGL_INFER_NUM_SAMPLERS_PER_NODE, -1);
  auto iface = GetEnv<std::string>(DGL_INFER_IFACE, "");

  auto input_trace_dir = GetEnv<std::string>(DGL_INFER_INPUT_TRACE_DIR, "");
  auto num_warmup_reqs = GetEnv<int>(DGL_INFER_NUM_WARMUPS, 1);
  auto num_reqs = GetEnv<int>(DGL_INFER_NUM_REQUESTS, 1);
  auto result_dir = GetEnv<std::string>(DGL_INFER_RESULT_DIR, "");
  auto collect_stats = GetEnv<bool>(DGL_INFER_COLLECT_STATS, false);
  auto execute_one_by_one = GetEnv<bool>(DGL_INFER_EXECUTE_ONE_BY_ONE, false);

  if (collect_stats) {
    EnableTracing();
  }

  auto parallel_type = GetEnumEnv<ParallelizationType>(DGL_INFER_PARALLELIZATION_TYPE);
  auto using_precomputed_aggregations = GetEnv<bool>(DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS, false);
  // TODO: env variables validation

  auto init_mon_actor = system.spawn<init_monitor_actor>();
  system.registry().put(caf::init_mon_atom_v, init_mon_actor);

  auto fin_mon_actor = system.spawn(fin_monitor_fn);
  system.registry().put(caf::fin_mon_atom_v, fin_mon_actor);

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

  auto exec_ctl_actor = system.spawn<executor_control_actor>(mpi_actor_ptr, num_devices_per_node);
  system.registry().put(caf::exec_control_atom_v, exec_ctl_actor);
  auto exec_ctl_actor_ptr = system.registry().get(caf::exec_control_atom_v);

  auto executor = spawn_executor_actor(
      system,
      parallel_type,
      exec_ctl_actor_ptr,
      mpi_actor_ptr,
      node_rank,
      num_nodes,
      num_backup_servers,
      num_devices_per_node,
      num_samplers_per_node,
      result_dir,
      execute_one_by_one,
      using_precomputed_aggregations);

  auto required_actors = std::vector<std::string>({ "mpi", "exec_ctrl" });
  auto result_receiver = system.spawn(result_receiver_fn, num_warmup_reqs, num_reqs);

  auto scheduler = system.spawn<scheduler_actor>(exec_ctl_actor_ptr,
                                                 result_receiver,
                                                 parallel_type,
                                                 using_precomputed_aggregations,
                                                 num_nodes,
                                                 num_devices_per_node,
                                                 num_samplers_per_node,
                                                 execute_one_by_one);
  caf::scoped_actor self { system };
  auto res_hdl = self->request(init_mon_actor, std::chrono::seconds(120), caf::wait_atom_v, required_actors);
  receive_result<bool>(res_hdl);

  std::cerr << "All services initialized." << std::endl;

  auto input_feader = system.spawn(input_feader_fn, scheduler, input_trace_dir, num_warmup_reqs, num_reqs);

  std::cout << "Wait for the warmup finished" << std::endl;
  auto warmup_rh = self->request(result_receiver, std::chrono::minutes(10), caf::wait_warmup_atom_v);
  receive_result<bool>(warmup_rh);

  std::cout << "Warmup done" << std::endl;

  self->send(input_feader, caf::start_atom_v);

  std::cout << "Wait for the experiment finished" << std::endl;
  auto exp_fin_rh = self->request(result_receiver, std::chrono::minutes(10), caf::wait_atom_v);
  receive_result<bool>(exp_fin_rh);

  if (collect_stats) {
    auto rh = self->request(exec_ctl_actor, caf::infinite, caf::write_trace_atom_v);
    receive_result<bool>(rh);
  }

  auto fin_rh = self->request(fin_mon_actor, caf::infinite, caf::done_atom_v);
  receive_result<bool>(fin_rh);

  self->send(input_feader, caf::done_atom_v);

  system.middleman().close(master_port);
  system.middleman().stop();
  // TODO: proper shutdown
  exit(0);
}

}
}
