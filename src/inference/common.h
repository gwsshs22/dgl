#pragma once

#include <caf/all.hpp>

#include <iostream>
#include <dgl/runtime/ndarray.h>

namespace dgl {
namespace inference {

using NDArray = dgl::runtime::NDArray;


template <typename T>
inline T receive_result(caf::response_handle<caf::blocking_actor, caf::message, true>& hdl) {
  T hold;
  hdl.receive(
    [&](T& res) { hold = std::move(res); },
    [&](caf::error& err) {
      // TODO: error handling.
      std::cerr << "Error : " << caf::to_string(err) << std::endl;
    }
  );

  return hold;
}

}
}
