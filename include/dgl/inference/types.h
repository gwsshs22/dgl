#pragma once

#include <dgl/runtime/ndarray.h>

namespace dgl {
namespace inference {

using NDArray = dgl::runtime::NDArray;

enum TaskType {
  kInitialize = 0,
  kInputBroadcast = 1,
  kTest = 999999,
};

}
}
