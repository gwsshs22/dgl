#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

class ObjectStorage {

 public:
  void CopyToSharedMemory(int batch_id, const std::string& name, const NDArray& src_arr);

  void Cleanup(int batch_id);

  static ObjectStorage* GetInstance() {
    static ObjectStorage storage;
    return &storage;
  }

  private:
   std::map<int, std::map<std::string, std::shared_ptr<runtime::SharedMemory>>> per_batch_metadata_holder_;
   std::map<int, std::map<std::string, NDArray>> per_batch_data_holder_;

   std::mutex mtx_;
};

}
}
