#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void input_bsend_fn(caf::blocking_actor* self,
                    const caf::actor& mpi_actor,
                    const NDArray& new_gnids,
                    const NDArray& new_features,
                    const NDArray& src_gnids,
                    const NDArray& dst_gnids,
                    const uint32_t tag);

void input_brecv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag);

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   int node_rank,
                   const NDArray& new_gnids,
                   const NDArray& new_features,
                   const NDArray& src_gnids,
                   const NDArray& dst_gnids,
                   const uint32_t tag);

void input_recv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag);

void move_input_to_shared_mem_fn(caf::blocking_actor *self,
                                 const caf::actor& object_storage_actor,
                                 const NDArray& new_gnids,
                                 const NDArray& new_features,
                                 const NDArray& src_gnids,
                                 const NDArray& dst_gnids);

// sampler
void sampling_fn(caf::blocking_actor *self,
                 const caf::actor& sampler,
                 int batch_id);

void cleanup_fn(caf::blocking_actor *self,
                const caf::actor& sampler,
                int batch_id);

enum gnn_executor_request_type {
  kComputeRequestType = 0,
  kCleanupRequestType = DGL_INFER_CLEANUP_REQUEST_TYPE
};

void gnn_execute_fn(caf::blocking_actor *self,
                    const caf::actor& gnn_executor_grp_actor,
                    int batch_id,
                    gnn_executor_request_type req_type,
                    int local_rank);

void gnn_broadcast_execute_fn(caf::blocking_actor *self,
                              const caf::actor& gnn_executor_grp_actor,
                              int batch_id,
                              gnn_executor_request_type req_type);

}
}
