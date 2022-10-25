#pragma once

#include <caf/all.hpp>
#include <caf/io/all.hpp>

namespace dgl {
namespace inference {

struct MpiCmdMsg {
};

template <class Inspector>
typename Inspector::result_type inspect(Inspector& f, MpiCmdMsg& x) {
  return f(caf::meta::type_name("MpiCmdMsg"));
}

}
}