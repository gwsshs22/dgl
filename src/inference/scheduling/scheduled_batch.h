#pragma once

#include <dgl/inference/common.h>

#include "scheduling.h"

namespace dgl {
namespace inference {

class ScheduledBatch {

 public:
  ScheduledBatch(int batch_id, BatchInput batch_input);

  ScheduledBatch(const ScheduledBatch& other) = delete;
  ScheduledBatch(ScheduledBatch&& other) = delete;

  enum Status {
    kCreated = 0,
    kInitializing = 1,
    kReady = 2,
    kRunning = 3,
    kFinished = 4
  };

  inline int batch_id() {
    return batch_id_;
  }

  inline Status status() {
    return status_;
  }

  inline const BatchInput& batch_input() {
    return batch_input_;
  }

  void SetStatus(const Status& to);

 private:
  const int batch_id_;
  BatchInput batch_input_;
  Status status_;
};

}
}

