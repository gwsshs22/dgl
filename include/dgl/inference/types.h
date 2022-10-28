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

enum SystemType {
  kMaster = 0,
  kWorker = 1,
  kActor = 2,
};

enum TaskType {
  kInitialize = 0,
  kInputBroadcast = 1,
  kTest = 999999,
};

}
}
