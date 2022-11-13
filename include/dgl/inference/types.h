#pragma once

#include <dgl/runtime/ndarray.h>

#define DGL_INFER_CLEANUP_REQUEST_TYPE 99999999

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
  kNone = 0,
  kInitialize = 1,
  kSampling = 2,
  kPrepareInput = 3,
  kCompute = 4,
  kPrepareAggregations = 5,
  kRecomputeAggregations = 6,
  kComputeRemaining = 7,
  kFetchResult = 15,
};

}
}
