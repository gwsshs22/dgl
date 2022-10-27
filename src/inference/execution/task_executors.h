#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void input_broadcast_actor(caf::blocking_actor* self,
                           const caf::actor& mpi_actor,
                           const NDArray& a,
                           const NDArray& b,
                           const NDArray& c);

void input_receive_actor(caf::blocking_actor* self, const caf::actor& mpi_actor);

}
}
