#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void input_bsend_all_fn(caf::blocking_actor* self,
                        const caf::actor& mpi_actor,
                        const caf::actor& object_storage_actor,
                        int batch_id,
                        const NDArray& new_gnids,
                        const NDArray& new_features,
                        const NDArray& src_gnids,
                        const NDArray& dst_gnids,
                        const uint32_t tag);

void input_brecv_all_fn(caf::blocking_actor* self,
                        const caf::actor& mpi_actor,
                        const caf::actor& object_storage_actor,
                        int batch_id,
                        const uint32_t tag);

void input_bsend_scatter_fn(caf::blocking_actor* self,
                            const caf::actor& mpi_actor,
                            const caf::actor& object_storage_actor,
                            int num_nodes,
                            BroadcastInitType init_type,
                            int batch_id,
                            const NDArray& new_gnids,
                            const NDArray& new_features,
                            const NDArray& src_gnids,
                            const NDArray& dst_gnids,
                            const FeatureSplitMethod& split_method,
                            const uint32_t tag);

void input_brecv_scatter_fn(caf::blocking_actor* self,
                            const caf::actor& mpi_actor,
                            const caf::actor& object_storage_actor,
                            int node_rank,
                            BroadcastInitType init_type,
                            int batch_id,
                            const uint32_t tag);

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   int node_rank,
                   int batch_id,
                   const NDArray& new_gnids,
                   const NDArray& new_features,
                   const NDArray& src_gnids,
                   const NDArray& dst_gnids,
                   const uint32_t tag);

void input_recv_fn(caf::blocking_actor* self,
                   const caf::actor& mpi_actor,
                   const caf::actor& object_storage_actor,
                   int batch_id,
                   const uint32_t tag);

void move_input_to_shared_mem_fn(caf::blocking_actor *self,
                                 const caf::actor& object_storage_actor,
                                 int batch_id,
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
  kP3OwnerComputeRequest = 1,
  kP3OtherComputeRequest = 2,
  kWriteTraces = DGL_INFER_WRITE_TRACES_REQUEST_TYPE,
  kCleanupRequestType = DGL_INFER_CLEANUP_REQUEST_TYPE
};

void gnn_execute_fn(caf::blocking_actor *self,
                    const caf::actor& gnn_executor_grp_actor,
                    int batch_id,
                    gnn_executor_request_type req_type,
                    int local_rank,
                    int param0);

void gnn_broadcast_execute_fn(caf::blocking_actor *self,
                              const caf::actor& gnn_executor_grp_actor,
                              int batch_id,
                              gnn_executor_request_type req_type,
                              int param0);

}
}
