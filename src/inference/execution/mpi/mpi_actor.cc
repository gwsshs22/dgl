#include "mpi_actor.h"

#include <dgl/runtime/ndarray.h>

#include "../mem_utils.h"

namespace dgl {
namespace inference {

void broadcast_send_(caf::blocking_actor* self,
                    caf::response_promise rp,
                    std::shared_ptr<GlooExecutor> gloo_executor,
                    u_int32_t rank,
                    const NDArray& data,
                    uint32_t tag);

void broadcast_recv_(caf::blocking_actor* self,
                     caf::response_promise rp,
                     std::shared_ptr<GlooExecutor> gloo_executor,
                     u_int32_t root_rank,
                     uint32_t tag);

void broadcast_recvsm_(caf::blocking_actor* self,
                       caf::response_promise rp,
                       std::shared_ptr<GlooExecutor> gloo_executor,
                       u_int32_t root_rank,
                       uint32_t tag,
                       int batch_id,
                       const std::string& name);

void send_(caf::blocking_actor* self,
           caf::response_promise rp,
           std::shared_ptr<GlooExecutor> gloo_executor,
           u_int32_t dst_rank,
           const NDArray& data,
           uint32_t tag);

void recv_(caf::blocking_actor* self,
           caf::response_promise rp,
           std::shared_ptr<GlooExecutor> gloo_executor,
           u_int32_t src_rank,
           uint32_t tag);

void recvsm_(caf::blocking_actor* self,
             caf::response_promise rp,
             std::shared_ptr<GlooExecutor> gloo_executor,
             u_int32_t src_rank,
             uint32_t tag,
             int batch_id,
             const std::string& name);

mpi_actor::mpi_actor(caf::actor_config& config,
          const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
          const MpiConfig& mpi_config)
    : event_based_actor(config),
      gloo_rendezvous_actor_ptr_(gloo_rendezvous_actor_ptr),
      mpi_config_(mpi_config) {
}

caf::behavior mpi_actor::make_behavior() {
  int rank = mpi_config_.rank;
  int num_nodes = mpi_config_.num_nodes;

  auto gloo_store = std::make_unique<ActorBasedStore>(this->system(), gloo_rendezvous_actor_ptr_);
  gloo_executor_ = std::make_shared<GlooExecutor>(std::move(gloo_store), rank, num_nodes);
  gloo_executor_->Initialize(mpi_config_.hostname, mpi_config_.iface);

  ReportToInitMon(*this, "mpi", rank, num_nodes);
  return {
    [=](caf::mpi_bsend_atom, const NDArray& data, uint32_t tag) {
      auto rp = make_response_promise<void>();
      spawn(broadcast_send_, rp, this->gloo_executor_, rank, data, tag);
      return rp;
    },
    [=](caf::mpi_brecv_atom, int root_rank, uint32_t tag) {
      auto rp = make_response_promise<NDArray>();
      spawn(broadcast_recv_, rp, this->gloo_executor_, root_rank, tag);
      return rp;
    },
    [=](caf::mpi_brecvsm_atom, int root_rank, uint32_t tag, int batch_id, const std::string& name) {
      auto rp = make_response_promise<NDArray>();
      spawn(broadcast_recvsm_, rp, this->gloo_executor_, root_rank, tag, batch_id, name);
      return rp;
    },
    [=](caf::mpi_send_atom, int dst_rank, const NDArray& data, uint32_t tag) {
      auto rp = make_response_promise<void>();
      spawn(send_, rp, this->gloo_executor_, dst_rank, data, tag);
      return rp;
    },
    [=](caf::mpi_recv_atom, int src_rank, uint32_t tag) {
      auto rp = make_response_promise<NDArray>();
      spawn(recv_, rp, this->gloo_executor_, src_rank, tag);
      return rp;
    },
    [=](caf::mpi_recvsm_atom, int src_rank, uint32_t tag, int batch_id, const std::string& name) {
      auto rp = make_response_promise<NDArrayWithSharedMeta>();
      spawn(recvsm_, rp, this->gloo_executor_, src_rank, tag, batch_id, name);
      return rp;
    },
  };
}

void broadcast_send_(caf::blocking_actor* self,
                    caf::response_promise rp,
                    std::shared_ptr<GlooExecutor> gloo_executor,
                    u_int32_t rank,
                    const NDArray& data,
                    uint32_t tag) {
  u_int8_t metadata_buf[__METADATA_MAX_BYTES];
  dmlc::MemoryFixedSizeStream strm(metadata_buf, __METADATA_MAX_BYTES);
  WriteMetadata(strm, data);
  gloo_executor->Broadcast(metadata_buf, __METADATA_MAX_BYTES, rank, tag);
  gloo_executor->Broadcast(data.Ptr<u_int8_t>(), data.GetSize(), rank, tag);

  rp.deliver(true);
}

void broadcast_recv_(caf::blocking_actor* self,
                    caf::response_promise rp,
                    std::shared_ptr<GlooExecutor> gloo_executor,
                    u_int32_t root_rank,
                    uint32_t tag) {
  u_int8_t metadata_buf[__METADATA_MAX_BYTES];
  gloo_executor->Broadcast(metadata_buf, __METADATA_MAX_BYTES, root_rank, tag);
  dmlc::MemoryFixedSizeStream strm(metadata_buf, __METADATA_MAX_BYTES);
  auto empty_arr = CreateEmptyArrayFromMetadata(strm);
  gloo_executor->Broadcast(empty_arr.Ptr<u_int8_t>(), empty_arr.GetSize(), root_rank, tag);

  rp.deliver(std::move(empty_arr));
}

void broadcast_recvsm_(caf::blocking_actor* self,
                       caf::response_promise rp,
                       std::shared_ptr<GlooExecutor> gloo_executor,
                       u_int32_t root_rank,
                       uint32_t tag,
                       int batch_id,
                       const std::string& name) {
  auto metadata = std::make_shared<runtime::SharedMemory>(GetArrayMetadataName(batch_id, name));
  auto buf = reinterpret_cast<u_int8_t*>(metadata->CreateNew(__METADATA_MAX_BYTES));
  gloo_executor->Broadcast(buf, __METADATA_MAX_BYTES, root_rank, tag);
  dmlc::MemoryFixedSizeStream strm(buf, __METADATA_MAX_BYTES);
  auto shared_arr = CreateEmptySharedArrayFromMetadata(strm, batch_id, name);
  gloo_executor->Broadcast(shared_arr.Ptr<u_int8_t>(), shared_arr.GetSize(), root_rank, tag);

  rp.deliver(std::make_pair(std::move(shared_arr), std::move(metadata)));
}

void send_(caf::blocking_actor* self,
          caf::response_promise rp,
          std::shared_ptr<GlooExecutor> gloo_executor,
          u_int32_t dst_rank,
          const NDArray& data,
          uint32_t tag) {
  u_int8_t metadata_buf[__METADATA_MAX_BYTES];
  dmlc::MemoryFixedSizeStream strm(metadata_buf, __METADATA_MAX_BYTES);
  WriteMetadata(strm, data);
  gloo_executor->Send(metadata_buf, __METADATA_MAX_BYTES, dst_rank, tag);
  gloo_executor->Send(data.Ptr<u_int8_t>(), data.GetSize(), dst_rank, tag);

  rp.deliver(true);
}

void recv_(caf::blocking_actor* self,
          caf::response_promise rp,
          std::shared_ptr<GlooExecutor> gloo_executor,
          u_int32_t src_rank,
          uint32_t tag) {
  u_int8_t metadata_buf[__METADATA_MAX_BYTES];
  gloo_executor->Recv(metadata_buf, __METADATA_MAX_BYTES, src_rank, tag);
  dmlc::MemoryFixedSizeStream strm(metadata_buf, __METADATA_MAX_BYTES);
  auto empty_arr = CreateEmptyArrayFromMetadata(strm);
  gloo_executor->Recv(empty_arr.Ptr<u_int8_t>(), empty_arr.GetSize(), src_rank, tag);

  rp.deliver(std::move(empty_arr));
}

void recvsm_(caf::blocking_actor* self,
             caf::response_promise rp,
             std::shared_ptr<GlooExecutor> gloo_executor,
             u_int32_t src_rank,
             uint32_t tag,
             int batch_id,
             const std::string& name) {
  auto metadata = std::make_shared<runtime::SharedMemory>(GetArrayMetadataName(batch_id, name));
  auto buf = reinterpret_cast<u_int8_t*>(metadata->CreateNew(__METADATA_MAX_BYTES));
  gloo_executor->Recv(buf, __METADATA_MAX_BYTES, src_rank, tag);
  dmlc::MemoryFixedSizeStream strm(buf, __METADATA_MAX_BYTES);
  auto shared_arr = CreateEmptySharedArrayFromMetadata(strm, batch_id, name);
  gloo_executor->Recv(shared_arr.Ptr<u_int8_t>(), shared_arr.GetSize(), src_rank, tag);

  rp.deliver(std::make_pair(std::move(shared_arr), std::move(metadata)));
}

}
}
