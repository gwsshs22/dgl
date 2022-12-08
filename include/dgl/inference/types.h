#pragma once

#include <dgl/runtime/ndarray.h>

#define DGL_INFER_WRITE_TRACES_REQUEST_TYPE 1000
#define DGL_INFER_CLEANUP_REQUEST_TYPE 99999999

namespace dgl {
namespace inference {

using NDArray = dgl::runtime::NDArray;
using NDArrayWithSharedMeta = std::pair<NDArray, std::shared_ptr<runtime::SharedMemory>>;
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

enum BroadcastInitType {
  kAll = 0,
  kScatter = 1,
  kScatterFeatureOnly = 1,
};

enum TaskType {
  kNone = 0,
  kInitialize = 1,
  kSampling = 2,
  kPushComputationGraph = 3,
  kCompute = 4,
  kFetchResult = 5,
  kCleanup = 6,
};

struct FeatureSplitMethod {
  int split_dimension;
  std::vector<int> split;
};

struct RequestStats {
  std::chrono::time_point<std::chrono::steady_clock> enqueued_time;

  RequestStats(const std::chrono::time_point<std::chrono::steady_clock>& enqueue_time_point) : enqueued_time(enqueue_time_point) {}

  RequestStats() = default;
  RequestStats(const RequestStats& other) = default;

  int ElapsedTimeInMicros() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - enqueued_time).count();
  }
};

}
}
