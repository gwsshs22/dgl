#include "mpi_actor.h"

#include <dlpack/dlpack.h>
#include <dmlc/memory_io.h>

#include <dgl/runtime/ndarray.h>

#include "gloo_rendezvous_actor.h"
#include "gloo_executor.h"

namespace dgl {
namespace inference {

inline void broadcast_(GlooExecutor& gloo_executor,
                       u_int32_t rank,
                       const NDArray& data);

inline NDArray receive_(GlooExecutor& gloo_executor,
                        u_int32_t root_rank);

mpi_actor::mpi_actor(caf::actor_config& config,
          const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
          const MpiConfig& mpi_config)
    : blocking_actor(config),
      gloo_rendezvous_actor_ptr_(gloo_rendezvous_actor_ptr),
      mpi_config_(mpi_config) {
}

void mpi_actor::act() {
  int rank = mpi_config_.rank;
  int num_nodes = mpi_config_.num_nodes;

  auto gloo_store = std::make_unique<ActorBasedStore>(this->system(), gloo_rendezvous_actor_ptr_);
  auto gloo_executor = GlooExecutor(std::move(gloo_store), rank, num_nodes);
  gloo_executor.Initialize(mpi_config_.hostname, mpi_config_.iface);

  bool running = true;
  ReportToInitMon(*this, "mpi", rank, num_nodes);
  receive_while([&]{ return running; }) (
    [&](caf::broadcast_atom, const NDArray& data) {
      broadcast_(gloo_executor, rank, data);
    },
    [&](caf::receive_atom, int root_rank) {
      return receive_(gloo_executor, root_rank);
    }
  );
}

constexpr unsigned int __METADATA_MAX_BYTES = 512;

inline void wirte_metadata_(dmlc::Stream& stream,
                            const NDArray& data) {
  stream.Write(data->ndim);
  stream.WriteArray(data->shape, data->ndim);
  stream.Write(data->dtype.code);
  stream.Write(data->dtype.bits);
  stream.Write(data->dtype.lanes);
}

inline NDArray create_from_metadata_(dmlc::Stream& stream) {
  int ndim;
  int64_t *shape;
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;

  stream.Read(&ndim);
  shape = new int64_t[ndim];
  stream.ReadArray(shape, ndim);
  stream.Read(&code);
  stream.Read(&bits);
  stream.Read(&lanes);

  auto shape_vector = std::vector<int64_t>();
  for (auto itr = shape; itr != shape + ndim; itr++) {
    shape_vector.push_back(*itr);
  }

  delete shape;
  
  return NDArray::Empty(shape_vector,
      DLDataType{code, bits, lanes},
      DLContext{kDLCPU, 0});
}

inline void broadcast_(GlooExecutor& gloo_executor,
                u_int32_t rank,
                const NDArray& data) {
  u_int8_t metadata_buf[__METADATA_MAX_BYTES];
  dmlc::MemoryFixedSizeStream strm(metadata_buf, __METADATA_MAX_BYTES);
  wirte_metadata_(strm, data);
  gloo_executor.Broadcast(metadata_buf, __METADATA_MAX_BYTES, rank);
  gloo_executor.Broadcast(data.Ptr<u_int8_t>(), data.GetSize(), rank);
}

inline NDArray receive_(GlooExecutor& gloo_executor,
                        u_int32_t root_rank) {
  u_int8_t metadata_buf[__METADATA_MAX_BYTES];
  gloo_executor.Broadcast(metadata_buf, __METADATA_MAX_BYTES, root_rank);
  dmlc::MemoryFixedSizeStream strm(metadata_buf, __METADATA_MAX_BYTES);
  auto empty_arr = create_from_metadata_(strm);
  gloo_executor.Broadcast(empty_arr.Ptr<u_int8_t>(), empty_arr.GetSize(), root_rank);

  return empty_arr;
}

}
}
