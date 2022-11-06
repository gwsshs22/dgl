#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void input_bsend_fn(caf::blocking_actor* self,
                    const caf::actor& mpi_actor,
                    const NDArray& new_ngids,
                    const NDArray& src_ngids,
                    const NDArray& dst_ngids);

void input_brecv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor);

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   int node_rank,
                   const NDArray& new_ngids,
                   const NDArray& src_ngids,
                   const NDArray& dst_ngids);

void input_recv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor);

}
}
