#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void input_bsend_fn(caf::blocking_actor* self,
                    const caf::actor& mpi_actor,
                    const NDArray& new_ngids,
                    const NDArray& src_ngids,
                    const NDArray& dst_ngids,
                    const uint32_t tag);

void input_brecv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag);

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   int node_rank,
                   const NDArray& new_ngids,
                   const NDArray& src_ngids,
                   const NDArray& dst_ngids,
                   const uint32_t tag);

void input_recv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag);

void move_input_to_shared_mem_fn(caf::blocking_actor *self,
                                 const caf::actor& object_storage_actor,
                                 const NDArray& new_ngids,
                                 const NDArray& src_ngids,
                                 const NDArray& dst_ngids);

void sampling_fn(caf::blocking_actor *self,
                 const caf::actor& sampler,
                 int batch_id);

void cleanup_fn(caf::blocking_actor *self,
                const caf::actor& sampler,
                int batch_id);

}
}
