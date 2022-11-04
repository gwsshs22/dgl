#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

struct BatchInput {
  NDArray new_gnids;
  NDArray src_gnids;
  NDArray dst_gnids;
};

class Scheduler {

 public:
  virtual void Initialize() = 0;
  virtual void BroadcastInitialize() = 0;
  virtual void Execute() = 0;
  virtual void BroadcastExecute() = 0;

};

class SchedulingPolicy {

 public:
  virtual void OnNewBatch(Scheduler& scheduler, BatchInput&& input) = 0;
  virtual void OnInitializeDone(Scheduler& scheduler) = 0;
  virtual void OnBatchInitializeDone(Scheduler& scheduler) = 0;
  virtual void OnExecuteDone(Scheduler& scheduler) = 0;
  virtual void OnBroadcastExecuteDone(Scheduler& scheduler) = 0;

};

}
}
