#include <iostream>
#include <cuda_runtime.h>
#include <utility>

#include <dgl/base_heterograph.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../../graph/transform/cuda/cuda_map_edges.cuh"
#include "../../array/cuda/atomic.cuh"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::runtime::cuda;
using namespace dgl::transform::cuda;
using namespace dgl::aten;

namespace omega {

std::pair<HeteroGraphPtr, IdArray> ToBlockGPU(const IdArray& u,
                                              const IdArray& v,
                                              const IdArray& dst_ids,
                                              const IdArray& src_ids,
                                              const IdArray& new_lhs_ids_prefix) {
  bool force_uncached = getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  const bool generate_src_ids = src_ids->shape[0] == 0;
  const auto& ctx = u->ctx;
  auto device = DeviceAPI::Get(ctx);
  cudaStream_t stream = getCurrentCUDAStream();

  int64_t max_rhs_nodes = dst_ids->shape[0];
  int64_t max_lhs_nodes;
  if (generate_src_ids) {
    max_lhs_nodes = new_lhs_ids_prefix->shape[0] + u->shape[0];
  } else {
    max_lhs_nodes = src_ids->shape[0];
  }

  int scale = 3;
  if (force_uncached) {
    scale = 2;
  }

  auto lhs_hash_table = std::make_unique<OrderedHashTable<int64_t>>(max_lhs_nodes, ctx, stream, scale);
  auto rhs_hash_table = std::make_unique<OrderedHashTable<int64_t>>(max_rhs_nodes, ctx, stream, scale);

  rhs_hash_table->FillWithUnique(dst_ids.Ptr<int64_t>(), max_rhs_nodes, stream);

  int64_t num_src_nodes = 0, num_dst_nodes = max_rhs_nodes;
  IdArray unique_src_nodes;

  if (generate_src_ids) {
    IdArray src_nodes = NewIdArray(max_lhs_nodes, ctx, sizeof(int64_t) * 8);
    int64_t src_nodes_offset = sizeof(int64_t) * new_lhs_ids_prefix->shape[0];
    device->CopyDataFromTo(
      new_lhs_ids_prefix.Ptr<int64_t>(), 0, src_nodes.Ptr<int64_t>(), 0,
      sizeof(int64_t) * new_lhs_ids_prefix->shape[0], new_lhs_ids_prefix->ctx,
      src_nodes->ctx, new_lhs_ids_prefix->dtype);

    device->CopyDataFromTo(
      u.Ptr<int64_t>(), 0, src_nodes.Ptr<int64_t>(), src_nodes_offset,
      sizeof(int64_t) * u->shape[0], u->ctx,
      src_nodes->ctx, u->dtype);

    IdArray lhs_nodes = NewIdArray(max_lhs_nodes, ctx, sizeof(int64_t) * 8);
    int64_t* count_lhs_device = static_cast<int64_t*>(
        device->AllocWorkspace(ctx, sizeof(int64_t)));
    
    CUDA_CALL(cudaMemsetAsync(
        count_lhs_device, 0, sizeof(*count_lhs_device), stream));

    lhs_hash_table->FillWithDuplicates(
      src_nodes.Ptr<int64_t>(), src_nodes->shape[0], lhs_nodes.Ptr<int64_t>(), count_lhs_device, stream
    );

    device->CopyDataFromTo(
        count_lhs_device, 0, &num_src_nodes, 0,
        sizeof(num_src_nodes), ctx,
        DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
    device->StreamSync(ctx, stream);

    // wait for the node counts to finish transferring
    device->FreeWorkspace(ctx, count_lhs_device);

    unique_src_nodes = NewIdArray(num_src_nodes, ctx, sizeof(int64_t) * 8);

    device->CopyDataFromTo(
      lhs_nodes.Ptr<int64_t>(), 0, unique_src_nodes.Ptr<int64_t>(), 0,
      sizeof(int64_t) * num_src_nodes, lhs_nodes->ctx,
      unique_src_nodes->ctx, lhs_nodes->dtype);
  } else {
    lhs_hash_table->FillWithUnique(src_ids.Ptr<int64_t>(), max_lhs_nodes, stream);
    num_src_nodes = max_lhs_nodes;
    unique_src_nodes = src_ids;
  }

  IdArray new_u;
  IdArray new_v;
  std::tie(new_u, new_v) = MapEdges(*lhs_hash_table, *rhs_hash_table, u, v, stream);

  std::vector<int64_t> num_nodes_per_type(2);
  std::vector<HeteroGraphPtr> rel_graphs;
  rel_graphs.reserve(1);

  const auto meta_graph = ImmutableGraph::CreateFromCOO(
    2,
    NDArray::FromVector(std::vector<int64_t>({0})),
    NDArray::FromVector(std::vector<int64_t>({1})));

  num_nodes_per_type[0] = num_src_nodes;
  num_nodes_per_type[1] = num_dst_nodes;

  rel_graphs.push_back(CreateFromCOO(
    2, num_src_nodes, num_dst_nodes,
    new_u, new_v));

  HeteroGraphPtr new_graph =
      CreateHeteroGraph(meta_graph, rel_graphs, num_nodes_per_type);

  return std::make_pair(new_graph, unique_src_nodes);
}


__global__ void _SplitEdgesKernel(
  const int64_t* ranges,
  const int num_parts,
  const int64_t num_total_edges,
  const int64_t* u,
  const int64_t* v,
  int64_t** new_srcs,
  int64_t** new_dsts,
  int32_t* num_edges_per_part) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_total_edges) {
    int64_t u_id = u[i];
    for (int part_id = 0; part_id < num_parts; part_id++) {
      if (u_id <= ranges[part_id]) {
        int new_idx = atomicAdd(num_edges_per_part + part_id, 1);
        new_srcs[part_id][new_idx] = u_id;
        new_dsts[part_id][new_idx] = v[i];
        break;
      }
    }
  }
}

__global__ void _CountEdges(
  const int64_t* ranges,
  const int num_parts,
  const int64_t num_total_edges,
  const int64_t* u,
  int32_t* num_edges_per_part) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_total_edges) {
    int64_t u_id = u[i];
    for (int part_id = 0; part_id < num_parts; part_id++) {
      if (u_id <= ranges[part_id]) {
        atomicAdd(num_edges_per_part + part_id, 1);
        break;
      }
    }
  }
}

std::pair<std::vector<IdArray>, std::vector<IdArray>> SplitEdgesGPU(
  const IdArray& target_gnids,
  const std::vector<int64_t>& num_assigned_targets,
  const IdArray& u,
  const IdArray& v) {
  CHECK_EQ(target_gnids->ctx.device_type, kDGLCUDA);
  CHECK_EQ(u->ctx.device_type, kDGLCUDA);
  CHECK_EQ(v->ctx.device_type, kDGLCUDA);

  const auto& ctx = u->ctx;
  auto device = DeviceAPI::Get(ctx);
  cudaStream_t stream = getCurrentCUDAStream();

  const int64_t num_total_edges = u->shape[0];
  const int num_parts = num_assigned_targets.size();

  std::vector<std::unique_ptr<OrderedHashTable<int64_t>>> hash_tables;

  IdArray ranges = NewIdArray({num_parts}, target_gnids->ctx, target_gnids->dtype.bits);
  IdArray num_edges_per_part = Full((int32_t)0, num_parts, target_gnids->ctx);

  int64_t offset = -1;
  for (int i = 0; i < num_parts; i++) {
    offset += num_assigned_targets[i];
    CHECK(offset < target_gnids->shape[0]);
    device->CopyDataFromTo(
      target_gnids.Ptr<int64_t>(), offset * sizeof(int64_t), ranges.Ptr<int64_t>(), i * sizeof(int64_t), sizeof(int64_t),
      target_gnids->ctx, ranges->ctx, target_gnids->dtype);
    
  }

  int64_t block_size = 128;
  int64_t num_blocks = (num_total_edges + block_size - 1) / block_size;

  CUDA_KERNEL_CALL(
      _CountEdges, num_blocks, block_size, 0, stream,
      ranges.Ptr<int64_t>(),
      num_parts,
      num_total_edges,
      u.Ptr<int64_t>(),
      num_edges_per_part.Ptr<int32_t>());

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));
  std::vector<int32_t> new_lens(num_parts);

  device->CopyDataFromTo(
      num_edges_per_part.Ptr<int32_t>(), 0, new_lens.data(), 0, num_parts * sizeof(int), ctx,
      DGLContext{kDGLCPU, 0}, DGLDataType{ kDGLInt, 32, 1 });
  CUDA_CALL(cudaEventRecord(copyEvent));
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  std::vector<IdArray> srcs_per_part, dsts_per_part;
  std::vector<int64_t*> srcs_ptrs, dsts_ptrs;

  int64_t** src_ptrs_device = static_cast<int64_t**>(device->AllocWorkspace(ctx, sizeof(int64_t*) * num_parts));
  int64_t** dst_ptrs_device = static_cast<int64_t**>(device->AllocWorkspace(ctx, sizeof(int64_t*) * num_parts));
  
  int64_t total_len = 0;
  for (int i = 0; i < num_parts; i++) {
    total_len += new_lens[i];
    srcs_per_part.push_back(IdArray::Empty({ new_lens[i] }, u->dtype, u->ctx));
    dsts_per_part.push_back(IdArray::Empty({ new_lens[i] }, u->dtype, u->ctx));

    srcs_ptrs.push_back(srcs_per_part[i].Ptr<int64_t>());
    dsts_ptrs.push_back(dsts_per_part[i].Ptr<int64_t>());
  }

  device->CopyDataFromTo(srcs_ptrs.data(), 0, src_ptrs_device, 0, sizeof(int64_t*) * num_parts, DGLContext{kDGLCPU}, ctx, u->dtype);
  device->CopyDataFromTo(dsts_ptrs.data(), 0, dst_ptrs_device, 0, sizeof(int64_t*) * num_parts, DGLContext{kDGLCPU}, ctx, u->dtype);

  CHECK_EQ(total_len, num_total_edges);

  num_edges_per_part = Full((int32_t)0, num_parts, target_gnids->ctx);

  CUDA_KERNEL_CALL(
      _SplitEdgesKernel, num_blocks, block_size, 0, stream,
      ranges.Ptr<int64_t>(),
      num_parts,
      num_total_edges,
      u.Ptr<int64_t>(),
      v.Ptr<int64_t>(),
      src_ptrs_device,
      dst_ptrs_device,
      num_edges_per_part.Ptr<int32_t>());

  return std::make_pair(srcs_per_part, dsts_per_part);
}

};
};
