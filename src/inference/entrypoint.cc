/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/inference/exec.h
 * \brief 
 */
#include "entrypoint.h"

#include <thread>

#include <dgl/inference/common.h>

#include "./process/master_process.h"
#include "./process/worker_process.h"
#include "./process/actor_process.h"

namespace dgl {
namespace inference {

#define DGL_INFERENCE_CAF_MAIN(...)                                                   \
  int dgl_inference_main(int argc, char** argv) {                                     \
    return caf::exec_main<__VA_ARGS__>(dgl::inference::caf_main, argc, argv);         \
  }

int dgl_inference_main(int argc, char** argv);

void ExecMasterProcess() {
  char *argv[] = {"", "-t", "0"};
  dgl_inference_main(3, argv);
}

void ExecWorkerProcess() {
  char *argv[] = {"", "-t", "1"};
  dgl_inference_main(3, argv);
}

void StartActorProcessThread() {
  auto thread = std::thread([]{
    char *argv[] = {"", "-t", "2"};
    dgl_inference_main(3, argv);
  });

  thread.detach();
}

void caf_main(caf::actor_system& system, const config& cfg) {
  if (cfg.type == SystemType::kMaster) {
    MasterProcessMain(system, cfg);
  } else if (cfg.type == SystemType::kWorker) {
    WorkerProcessMain(system, cfg);
  } else {
    ActorProcessMain(system, cfg);
  }
}

DGL_INFERENCE_CAF_MAIN(caf::id_block::core_extension, caf::io::middleman)

}  // namespace inference
}  // namespace dgl
