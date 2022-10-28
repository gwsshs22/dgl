#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void ActorProcessMain(caf::actor_system& system, const config& cfg);

}  // namespace inference
}  // namespace dgl
