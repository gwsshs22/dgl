#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void MasterProcessMain(caf::actor_system& system, const config& cfg);

}  // namespace inference
}  // namespace dgl
