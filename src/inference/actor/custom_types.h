#pragma once

#include <caf/all.hpp>
#include <caf/io/all.hpp>

namespace dgl {
namespace inference {



struct MpiInitMsg {
  caf::strong_actor_ptr actor_ptr;
  int rank = -1;
  int world_size = -1;
};

template <class Inspector>
typename Inspector::result_type inspect(Inspector& f, MpiInitMsg& x) {
  return f(caf::meta::type_name("MpiInitMsg"), x.actor_ptr, x.rank, x.world_size);
}

struct MpiCmdMsg {
};

template <class Inspector>
typename Inspector::result_type inspect(Inspector& f, MpiCmdMsg& x) {
  return f(caf::meta::type_name("MpiCmdMsg"));
}

}
}