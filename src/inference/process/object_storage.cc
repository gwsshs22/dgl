#include "object_storage.h"

#include "../execution/mem_utils.h"

namespace dgl {
namespace inference {

void ObjectStorage::CopyToSharedMemory(int batch_id, const std::string& name, const NDArray& src_arr) {
  auto metadata_name = GetArrayMetadataName(batch_id, name);
  assert(!runtime::SharedMemory::Exist(metadata_name));
  auto metadata = CreateMetadataSharedMem(metadata_name, src_arr);

  // It turns out that it is impossible to share device memory between processes.
  assert(src_arr->ctx.device_type == kDLCPU);
  auto copied = CopyToSharedMem(batch_id, name, src_arr);

  {
    std::lock_guard<std::mutex> lock(mtx_);
    per_batch_metadata_holder_[batch_id][name] = metadata;
    per_batch_data_holder_[batch_id][name] = copied;
  }
}

void ObjectStorage::Cleanup(int batch_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = per_batch_metadata_holder_.find(batch_id);
  assert(it != per_batch_metadata_holder_.end());
  per_batch_metadata_holder_.erase(it);

  auto it2 = per_batch_data_holder_.find(batch_id);
  assert(it2 != per_batch_data_holder_.end());
  per_batch_data_holder_.erase(it2);
}

}
}
