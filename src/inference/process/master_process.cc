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

caf::behavior result_collector_fn(caf::event_based_actor* self) {
  return [&](caf::done_atom, int req_id, const NDArray& result) {
    const int num_inputs = 10;
    std::cout << "req_id=" << req_id << ", shape=(";
    for (int i = 0; i < result->ndim; i++) {
      std::cout<<result->shape[i] << ", ";
    }

    std::cout << ") " << result << std::endl;
  };
}

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

  auto executor = spawn_executor_actor(
      system,
      parallel_type,
      exec_ctl_actor_ptr,
      mpi_actor_ptr,
      node_rank,
      num_nodes,
      num_devices_per_node,
      using_precomputed_aggregations);

  auto required_actors = std::vector<std::string>({ "mpi", "exec_ctrl" });
  auto result_collector = system.spawn(result_collector_fn);

  auto scheduler = system.spawn<scheduler_actor>(exec_ctl_actor_ptr,
                                                 result_collector,
                                                 parallel_type,
                                                 using_precomputed_aggregations,
                                                 num_nodes,
                                                 num_devices_per_node);

  caf::scoped_actor self { system };
  auto res_hdl = self->request(init_mon_actor, std::chrono::seconds(60), caf::wait_atom_v, required_actors);
  receive_result<bool>(res_hdl);

  std::cerr << "All services initialized." << std::endl;

  auto cpu_context = DLContext { kDLCPU, 0 };

  int num_inputs = 10;
  int feature_size = 256;
  NDArray new_features = NDArray::Empty({num_inputs, feature_size}, DLDataType{kDLFloat, 32, 1}, cpu_context);
  float* ptr = (float*)new_features->data;
  for (int i = 0; i < num_inputs * feature_size; i++) {
    *ptr++ = (float)(i + 1) / (float)feature_size;
  }

  NDArray new_gnids = NDArray::FromVector(std::vector<int64_t>{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, cpu_context);
  
  NDArray src_gnids = NDArray::FromVector(std::vector<int64_t>{ 10, 10, 10, 10, 10,  3,  4,  0,  7,  9,  0, 12, 13, 14, 15, 9  }, cpu_context);
  NDArray dst_gnids = NDArray::FromVector(std::vector<int64_t>{  0,  1, 12, 13, 11, 10, 12, 13, 10, 10, 14, 15, 16, 17, 18, 19 }, cpu_context);

  for (int j = 0; j < 1; j++) {
    for (int i = 0; i < 1; i++) {
      anon_send(scheduler, caf::enqueue_atom_v, new_gnids, new_features, src_gnids, dst_gnids);
    }
  }

  std::cerr << "Quit to enter:" << std::endl;
  std::string dummy;
  std::getline(std::cin, dummy);

  exit(0);
}

}
}
