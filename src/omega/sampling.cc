#include "sampling.h"

#include <dmlc/omp.h>
#include <parallel_hashmap/phmap.h>

#include <dgl/random.h>
#include <dgl/runtime/parallel_for.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace omega {

typedef phmap::flat_hash_map<int64_t, int64_t> IdMap;

std::pair<IdArray, IdArray> _BuildCSRData(
  const IdArray& target_gnids,
  const IdArray& src_gnids,
  const IdArray& dst_gnids,
  IdMap& tid_to_rid) {

  const int64_t N = target_gnids->shape[0];
  const int64_t NNZ = src_gnids->shape[0];

  const int64_t* row_data = dst_gnids.Ptr<const int64_t>();
  const int64_t* col_data = src_gnids.Ptr<const int64_t>();

  NDArray ret_indptr = NDArray::Empty({N + 1}, src_gnids->dtype, src_gnids->ctx);
  NDArray ret_indices = NDArray::Empty({NNZ}, src_gnids->dtype, src_gnids->ctx);

  int64_t *const Bp = ret_indptr.Ptr<int64_t>();
  Bp[0] = 0;
  int64_t *const Bi = ret_indices.Ptr<int64_t>();

  // the offset within each row, that each thread will write to
  std::vector<std::vector<int64_t>> local_ptrs;
  std::vector<int64_t> thread_prefixsum;

#pragma omp parallel
  {
    const int num_threads = omp_get_num_threads();
    const int thread_id = omp_get_thread_num();
    CHECK_LT(thread_id, num_threads);

    const int64_t nz_chunk = (NNZ + num_threads - 1) / num_threads;
    const int64_t nz_start = thread_id * nz_chunk;
    const int64_t nz_end = std::min(NNZ, nz_start + nz_chunk);

    const int64_t n_chunk = (N + num_threads - 1) / num_threads;
    const int64_t n_start = thread_id * n_chunk;
    const int64_t n_end = std::min(N, n_start + n_chunk);

#pragma omp master
    {
      local_ptrs.resize(num_threads);
      thread_prefixsum.resize(num_threads + 1);
    }

#pragma omp barrier
    local_ptrs[thread_id].resize(N, 0);

    for (int64_t i = nz_start; i < nz_end; ++i) {
      ++local_ptrs[thread_id][tid_to_rid[row_data[i]]];
    }

#pragma omp barrier
    // compute prefixsum in parallel
    int64_t sum = 0;
    for (int64_t i = n_start; i < n_end; ++i) {
      int64_t tmp = 0;
      for (int j = 0; j < num_threads; ++j) {
        std::swap(tmp, local_ptrs[j][i]);
        tmp += local_ptrs[j][i];
      }
      sum += tmp;
      Bp[i + 1] = sum;
    }
    thread_prefixsum[thread_id + 1] = sum;

#pragma omp barrier
#pragma omp master
    {
      for (int64_t i = 0; i < num_threads; ++i) {
        thread_prefixsum[i + 1] += thread_prefixsum[i];
      }
      CHECK_EQ(thread_prefixsum[num_threads], NNZ);
    }
#pragma omp barrier

    sum = thread_prefixsum[thread_id];
    for (int64_t i = n_start; i < n_end; ++i) {
      Bp[i + 1] += sum;
    }

#pragma omp barrier
    for (int64_t i = nz_start; i < nz_end; ++i) {
      const int64_t r = tid_to_rid[row_data[i]];
      const int64_t index = Bp[r] + local_ptrs[thread_id][r]++;
      Bi[index] = col_data[i];
    }
  }
  CHECK_EQ(Bp[N], NNZ);

  return std::make_pair(ret_indptr, ret_indices);
}


inline int64_t _NumPicksFn(int64_t num_samples, bool replace, int64_t rowid, int64_t off, int64_t len, const int64_t* col) {
  const int64_t max_num_picks = (num_samples == -1) ? len : num_samples;
  if (replace) {
    return static_cast<int64_t>(len == 0 ? 0 : max_num_picks);
  } else {
    return std::min(static_cast<int64_t>(max_num_picks), len);
  }
}

inline void _PickFn(int64_t num_samples, bool replace, int64_t rowid, int64_t off,
                int64_t len, int64_t num_picks, const int64_t* col, int64_t* out_idx) {
  RandomEngine::ThreadLocal()->UniformChoice<int64_t>(
      num_picks, len, out_idx, replace);
  for (int64_t j = 0; j < num_picks; ++j) {
    out_idx[j] += off;
  }
}

std::pair<IdArray, IdArray> _SampleEdges(
    const IdArray& target_gnids,
    const IdArray& indptr_arr,
    const IdArray& indices_arr,
    int num_samples) {

  const int64_t num_rows = target_gnids->shape[0];
  const auto& ctx = indptr_arr->ctx;
  const auto& idtype = indptr_arr->dtype;

  const int64_t* tid_data = target_gnids.Ptr<int64_t>();
  const int64_t* indptr = indptr_arr.Ptr<int64_t>();
  const int64_t* indices = indices_arr.Ptr<int64_t>();

  const int num_threads = runtime::compute_num_threads(0, num_rows, 1);
  std::vector<int64_t> global_prefix(num_threads + 1, 0);

  IdArray picked_row, picked_col, picked_idx;
#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();

    const int64_t start_i =
        thread_id * (num_rows / num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_rows % num_threads);
    const int64_t end_i =
        (thread_id + 1) * (num_rows / num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_rows % num_threads);
    assert(thread_id + 1 < num_threads || end_i == num_rows);

    const int64_t num_local = end_i - start_i;

    // make sure we don't have to pay initialization cost
    std::unique_ptr<int64_t[]> local_prefix(new int64_t[num_local + 1]);
    local_prefix[0] = 0;
    for (int64_t i = start_i; i < end_i; ++i) {
      // build prefix-sum
      const int64_t local_i = i - start_i;
      const int64_t rid = i;
      int64_t len = _NumPicksFn(
          num_samples, false, rid, indptr[rid], indptr[rid + 1] - indptr[rid], indices);
      local_prefix[local_i + 1] = local_prefix[local_i] + len;
    }
    global_prefix[thread_id + 1] = local_prefix[num_local];

#pragma omp barrier
#pragma omp master
    {
      for (int t = 0; t < num_threads; ++t) {
        global_prefix[t + 1] += global_prefix[t];
      }
      picked_row = IdArray::Empty({global_prefix[num_threads]}, idtype, ctx);
      picked_col = IdArray::Empty({global_prefix[num_threads]}, idtype, ctx);
      picked_idx = IdArray::Empty({global_prefix[num_threads]}, idtype, ctx);
    }

#pragma omp barrier
    int64_t* picked_rdata = picked_row.Ptr<int64_t>();
    int64_t* picked_cdata = picked_col.Ptr<int64_t>();
    int64_t* picked_idata = picked_idx.Ptr<int64_t>();

    const int64_t thread_offset = global_prefix[thread_id];

    for (int64_t i = start_i; i < end_i; ++i) {
      const int64_t rid = i;

      const int64_t off = indptr[rid];
      const int64_t len = indptr[rid + 1] - off;
      if (len == 0) continue;

      const int64_t local_i = i - start_i;
      const int64_t row_offset = thread_offset + local_prefix[local_i];
      const int64_t num_picks =
          thread_offset + local_prefix[local_i + 1] - row_offset;

      _PickFn(
          num_samples, false, rid, off, len, num_picks, indices, picked_idata + row_offset);
      for (int64_t j = 0; j < num_picks; ++j) {
        const int64_t picked = picked_idata[row_offset + j];
        picked_rdata[row_offset + j] = tid_data[rid];
        picked_cdata[row_offset + j] = indices[picked];
      }
    }
  }

  const int64_t new_len = global_prefix.back();

  IdArray a, b;
  return std::make_pair(
    picked_col.CreateView({new_len}, picked_col->dtype),
    picked_row.CreateView({new_len}, picked_row->dtype));
}

std::vector<std::pair<IdArray, IdArray>> SampleEdges(
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const std::vector<int64_t>& fanouts) {

  IdMap tid_to_rid;
  const int64_t* t_data = target_gnids.Ptr<int64_t>();
  for (int i = 0; i < target_gnids->shape[0]; i++) {
    tid_to_rid.insert({ t_data[i], i });
  }

  auto csr_data = _BuildCSRData(target_gnids, src_gnids, dst_gnids, tid_to_rid);
  IdArray indptr = csr_data.first;
  IdArray indices = csr_data.second;

  std::vector<std::pair<IdArray, IdArray>> ret;

  for (const auto& fanout : fanouts) {
    ret.push_back(_SampleEdges(target_gnids, indptr, indices, fanout));
  }

  return ret;
}

}
}
