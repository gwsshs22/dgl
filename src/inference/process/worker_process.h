#pragma once

#include "process.h"

namespace dgl {
namespace inference {

void WorkerProcessMain(caf::actor_system& system, const config& cfg);

}  // namespace inference
}  // namespace dgl
