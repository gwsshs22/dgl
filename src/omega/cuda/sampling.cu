#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <utility>

#include <dgl/base_heterograph.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/device_api.h>
#include <dgl/random.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../../c_api_common.h"

#include <dgl/runtime/c_runtime_api.h>
#include "cuda_runtime.h"
#include "../../array/cuda/utils.h"
#include "../../array/cuda/atomic.cuh"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;
using namespace dgl::aten::cuda;
using namespace dgl::cuda;

namespace omega {

namespace {

constexpr int BLOCK_SIZE = 128;

template <typename T>
int _NumberOfBits(const T& range) {
  if (range <= 1) {
    // ranges of 0 or 1 require no bits to store
    return 0;
  }

  int bits = 1;
  while (bits < static_cast<int>(sizeof(T) * 8) && (1 << bits) < range) {
    ++bits;
  }

  CHECK_EQ((range - 1) >> bits, 0);
  CHECK_NE((range - 1) >> (bits - 1), 0);

  return bits;
}

__global__ void _SortedSearchKernelUpperBound(
    const int64_t* hay, int64_t hay_size, const int64_t* needles,
    int64_t num_needles, int64_t* pos) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    const int64_t ele = needles[tx];
    // binary search
    int64_t lo = 0, hi = hay_size;
    while (lo < hi) {
      int64_t mid = (lo + hi) >> 1;
      if (hay[mid] <= ele) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    pos[tx] = lo;
    tx += stride_x;
  }
}

__global__ void _CSRRowWiseSampleDegreeKernel(
    const int64_t num_picks, const int64_t num_rows,
    const int64_t* const in_rows, const int64_t* const in_ptr,
    int64_t* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(
        static_cast<int64_t>(num_picks), in_ptr[in_row + 1] - in_ptr[in_row]);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}


/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, without replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const in_index,
    const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
    IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_idxs[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = perm_idx;
      }
    }
    out_row += 1;
  }
}
}

std::pair<IdArray, IdArray> _SortCOO(
  const int64_t num_targets,
  const IdArray& src_gnids,
  const IdArray& dst_gnids) {

  // const int num_bits = _NumberOfBits(num_targets);

  auto sorted = Sort(dst_gnids);
  IdArray sorted_src_gnids = IndexSelect(src_gnids, sorted.second);

  return std::make_pair(sorted_src_gnids, sorted.first);
}

std::pair<IdArray, IdArray> _BuildCSRData(
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids) {
  const auto& ctx = target_gnids->ctx;
  const auto nbits = target_gnids->dtype.bits;
  const int64_t num_targets = target_gnids->shape[0];
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  auto sorted = _SortCOO(num_targets, src_gnids, dst_gnids);

  const IdArray& sorted_src_gnids = sorted.first;
  const IdArray& sorted_dst_gnids = sorted.second;

  const int64_t nnz = sorted_dst_gnids->shape[0];

  const int nt = dgl::cuda::FindNumThreads(num_targets);
  const int nb = (num_targets + nt - 1) / nt;
  IdArray indptr = Full(0, num_targets + 1, nbits, ctx);
  CUDA_KERNEL_CALL(
      _SortedSearchKernelUpperBound, nb, nt, 0, stream, sorted_dst_gnids.Ptr<int64_t>(),
      nnz, target_gnids.Ptr<int64_t>(), num_targets, indptr.Ptr<int64_t>() + 1);
  
  return std::make_pair(indptr, sorted_src_gnids);
}

std::pair<IdArray, IdArray> _SampleEdges(
    const IdArray& target_gnids,
    const IdArray& indptr,
    const IdArray& indices,
    int64_t num_picks) {
  const auto& ctx = target_gnids->ctx;
  const auto nbits = target_gnids->dtype.bits;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = target_gnids->shape[0];

  const IdArray slice_rows_arr = Range(0, num_rows, nbits, ctx);
  const int64_t* const slice_rows = static_cast<const int64_t*>(slice_rows_arr->data);

  IdArray picked_row =
      NewIdArray(num_rows * num_picks, ctx, sizeof(int64_t) * 8);
  IdArray picked_col =
      NewIdArray(num_rows * num_picks, ctx, sizeof(int64_t) * 8);
  IdArray picked_idx =
      NewIdArray(num_rows * num_picks, ctx, sizeof(int64_t) * 8);
  int64_t* const out_rows = static_cast<int64_t*>(picked_row->data);
  int64_t* const out_cols = static_cast<int64_t*>(picked_col->data);
  int64_t* const out_idxs = static_cast<int64_t*>(picked_idx->data);

  const int64_t* in_ptr = static_cast<int64_t*>(GetDevicePointer(indptr));
  const int64_t* in_cols = static_cast<int64_t*>(GetDevicePointer(indices));

  
  int64_t* out_deg = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(int64_t)));
  
  {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
        num_rows, slice_rows, in_ptr, out_deg);
  }
  
  // fill out_ptr
  int64_t* out_ptr = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(int64_t)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  // TODO(dlasalle): use pinned memory to overlap with the actual sampling, and
  // wait on a cudaevent
  int64_t new_len;
  // copy using the internal current stream
  device->CopyDataFromTo(
      out_ptr, num_rows * sizeof(new_len), &new_len, 0, sizeof(new_len), ctx,
      DGLContext{kDGLCPU, 0}, indptr->dtype);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  
  // select edges
  // the number of rows each thread block will cover
  {
    constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformKernel<int64_t, TILE_SIZE>), grid, block, 0,
        stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
        out_ptr, out_rows, out_cols, out_idxs);
  }

  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);

  IdArray remapped_picked_row = IndexSelect(target_gnids, picked_row);

  return std::make_pair(picked_col, remapped_picked_row);
}

std::vector<std::pair<IdArray, IdArray>> SampleEdgesGPU(
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const std::vector<int64_t>& fanouts) {

  CHECK_EQ(target_gnids->ctx.device_type, DGLDeviceType::kDGLCUDA);
  CHECK_EQ(src_gnids->ctx.device_type, DGLDeviceType::kDGLCUDA);
  CHECK_EQ(dst_gnids->ctx.device_type, DGLDeviceType::kDGLCUDA);

  std::vector<std::pair<IdArray, IdArray>> ret;

  auto csr_data = _BuildCSRData(target_gnids, src_gnids, dst_gnids);

  IdArray indptr = csr_data.first;
  IdArray indices = csr_data.second;

  for (const auto& fanout : fanouts) {
    ret.push_back(_SampleEdges(target_gnids, indptr, indices, fanout));
  }

  return ret;
}

}
}