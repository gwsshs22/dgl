#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

class ExecutionContext {

 public:
  ExecutionContext(int batch_id);

  ExecutionContext(int batch_id, const NDArray& new_ngids, const NDArray& src_ngids, const NDArray& dst_ngids);

  ExecutionContext(const ExecutionContext& other) = delete;
  ExecutionContext(ExecutionContext&& other) = delete;

  void SetBatchInput(const NDArray& new_ngids,
                     const NDArray& src_ngids,
                     const NDArray& dst_ngids);

  inline const NDArray& new_ngids() {
    return new_ngids_;
  }

  inline const NDArray& src_ngids() {
    return src_ngids_;
  }

  inline const NDArray& dst_ngids() {
    return dst_ngids_;
  }

  // TODO: remove this
  void Print() {
    std::cerr << "batch_id = " << batch_id_ << ", " << new_ngids_ << ", " << src_ngids_ << ", " << dst_ngids_ << std::endl;
  }

 private:
  const int batch_id_;

  NDArray new_ngids_;
  NDArray src_ngids_;
  NDArray dst_ngids_;
};

}
}