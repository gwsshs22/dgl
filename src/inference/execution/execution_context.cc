#include "execution_context.h"

namespace dgl {
namespace inference {

ExecutionContext::ExecutionContext(int batch_id,
                                   int local_rank,
                                   const NDArray& new_ngids,
                                   const NDArray& src_ngids,
                                   const NDArray& dst_ngids)
    : batch_id_(batch_id),
      local_rank_(local_rank),
      new_ngids_(new_ngids),
      src_ngids_(src_ngids),
      dst_ngids_(dst_ngids) {
}

ExecutionContext::ExecutionContext(int batch_id, int local_rank) : batch_id_(batch_id), local_rank_(local_rank) {
}

void ExecutionContext::SetBatchInput(const NDArray& new_ngids,
                                     const NDArray& src_ngids,
                                     const NDArray& dst_ngids) {
  new_ngids_ = new_ngids;
  src_ngids_ = src_ngids;
  dst_ngids_ = dst_ngids;
}

}
}
