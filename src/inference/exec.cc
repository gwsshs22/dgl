/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/inference/exec.h
 * \brief 
 */
#include "actor/actor_types.h"
#include "exec.h"
#include "master_process.h"
#include "worker_process.h"

#include "mpi_actor.h"

namespace dgl {
namespace inference {

#define DGL_INFERENCE_CAF_MAIN(...)                                                   \
  int dgl_inference_main(int argc, char** argv) {                                     \
    return caf::exec_main<__VA_ARGS__>(dgl::inference::caf_main, argc, argv);         \
  }

int dgl_inference_main(int argc, char** argv);

void ExecMasterProcess() {
  char *argv[] = {"", "-s", "-p", "43207"};
  dgl_inference_main(4, argv);
}

void ExecWorkerProcess() {
  char *argv[] = {"", "-p", "43207"};
  dgl_inference_main(3, argv);
}

void caf_main(caf::actor_system& system, const config& cfg) {
  auto f = cfg.server_mode ? MasterProcessMain : WorkerProcessMain;
  f(system, cfg);
}

DGL_INFERENCE_CAF_MAIN(caf::id_block::core_extension, caf::io::middleman)

}  // namespace inference
}  // namespace dgl
