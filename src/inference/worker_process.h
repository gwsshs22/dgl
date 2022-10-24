#pragma once

#include "exec.h"

namespace dgl {
namespace inference {

void WorkerProcessMain(caf::actor_system& system, const config& cfg);

}  // namespace inference
}  // namespace dgl
