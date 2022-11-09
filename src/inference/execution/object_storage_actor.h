#pragma once

#include <dgl/runtime/shared_mem.h>

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

struct object_storage {
  int batch_id;
  std::unordered_map<std::string, NDArray> arrays;
  std::unordered_map<std::string, std::shared_ptr<runtime::SharedMemory>> metadata_shared_mems;
};

caf::behavior object_storage_actor(caf::stateful_actor<object_storage>* self,
                                   int batch_id);

}
}
