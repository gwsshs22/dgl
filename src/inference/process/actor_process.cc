#include "actor_process.h"

#include <thread>

#include <dgl/inference/envs.h>

#include "process_control_actor.h"

namespace dgl {
namespace inference {

void ActorProcessMain(caf::actor_system& system, const config& cfg) {
  auto master_host = GetEnv<std::string>(DGL_INFER_MASTER_HOST, "localhost");
  auto master_port = GetEnv<u_int16_t>(DGL_INFER_MASTER_PORT, 0);
  auto actor_process_global_id = GetEnv<int>(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, -1);
  auto role = GetEnv<std::string>(DGL_INFER_ACTOR_PROCESS_ROLE, "");
  auto local_rank = GetEnv<int>(DGL_INFER_LOCAL_RANK, -1);
  // TODO: env variables validation

  auto node = retry<caf::node_id>([&] {
    return system.middleman().connect(master_host, master_port);
  });

  if (!node) {
    // TODO: error handling
    std::cerr << "*** connect failed: " << caf::to_string(node.error()) << std::endl;
    return;
  }

  auto process_mon_actor_ptr = system.middleman().remote_lookup(caf::process_mon_atom_v, node.value());
  system.registry().put(caf::process_mon_atom_v, process_mon_actor_ptr);
  auto process_mon_actor = caf::actor_cast<caf::actor>(process_mon_actor_ptr);

  auto process_request_handler = system.spawn(process_request_handler_fn, actor_process_global_id);


  std::this_thread::sleep_for(std::chrono::seconds(30));

}

}
}
