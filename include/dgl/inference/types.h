#pragma once

#include <dgl/runtime/ndarray.h>

namespace dgl {
namespace inference {

using NDArray = dgl::runtime::NDArray;
using EnvSetter = std::function<void()>;

struct config : caf::actor_system_config {
  config() {
    opt_group{custom_options_, "global"}
      .add(type, "type,t", "system type");
  }
  int type;
};

enum ActorSystemType {
  kMasterProcess = 0,
  kWorkerProcess = 1,
  kActorProcess = 2,
};

enum TaskType {
  kInitialize = 0,
  kSampling = 1,
  kPrepareInput = 2,
  kTest = 999999,
};

}
}
