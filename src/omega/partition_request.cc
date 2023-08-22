#include "partition_request.h"

#include <dmlc/omp.h>
#include <algorithm>
#include <chrono>

#include <dgl/runtime/parallel_for.h>
#include <parallel_hashmap/phmap.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace omega {

typedef phmap::flat_hash_map<int64_t, int64_t> IdMap;

std::pair<std::vector<IdArray>, std::vector<IdArray>> PartitionRequest(
    int num_machines,
    int num_gpus_per_machine,
    const IdArray& nid_partitions,
    const IdArray& num_assigned_targets_per_gpu,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids) {

  int num_partitions = num_machines * num_gpus_per_machine;
  int num_targets = target_gnids->shape[0];
  int64_t num_edges = src_gnids->shape[0];

  const int64_t* nid_partitions_data = static_cast<int64_t*>(nid_partitions->data);
  const int64_t* num_assigned_targets_per_gpu_data = static_cast<int64_t*>(num_assigned_targets_per_gpu->data);
  const int64_t* target_gnids_data = static_cast<int64_t*>(target_gnids->data);
  const int64_t* src_gnids_data = static_cast<int64_t*>(src_gnids->data);
  const int64_t* dst_gnids_data = static_cast<int64_t*>(dst_gnids->data);

  IdMap target_id_to_part_id;

  int i = 0;
  for (int part_id = 0; part_id < num_partitions; part_id++) {
    for (int j = 0; j < num_assigned_targets_per_gpu_data[part_id]; j++) {
      target_id_to_part_id.insert({ target_gnids_data[i], part_id });
      i++;
    }
  }

  CHECK_EQ(target_id_to_part_id.size(), num_targets);

  std::vector<IdArray> src_gnids_list(num_partitions);
  std::vector<IdArray> dst_gnids_list(num_partitions);
  std::vector<int64_t*> src_gnids_ptr_list(num_partitions, nullptr);
  std::vector<int64_t*> dst_gnids_ptr_list(num_partitions, nullptr);

  const int num_threads = runtime::compute_num_threads(0, num_edges, 1);
  std::vector<std::vector<int64_t>> p_sum;
  p_sum.resize(num_threads + 1);
  p_sum[0].resize(num_partitions, 0);
  std::unique_ptr<int[]> part_data(new int[num_edges]);
#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();
    const int64_t start_i =
        thread_id * (num_edges / num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_edges % num_threads);
    const int64_t end_i =
        (thread_id + 1) * (num_edges / num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_edges % num_threads);
    assert(thread_id + 1 < num_threads || end_i == num_edges);

    p_sum[thread_id + 1].resize(num_partitions, 0);

    for (int64_t i = start_i; i < end_i; i++) {
      int part_id = -1;
      auto src_gnid = src_gnids_data[i];

      for (int j = 0; j < num_machines; j++) {
        if (src_gnid < nid_partitions_data[j + 1]) {
          part_id = j * num_gpus_per_machine + src_gnid % num_gpus_per_machine;
          break;
        }
      }

      if (part_id == -1) {
        part_id = target_id_to_part_id[src_gnid];
      }

      part_data[i] = part_id;
      p_sum[thread_id + 1][part_id]++;
    }

#pragma omp barrier
#pragma omp master
    {
      int64_t cumsum = 0;
      for (int p = 0; p < num_partitions; p++) {
        for (int j = 0; j < num_threads; j++) {
          p_sum[j + 1][p] += p_sum[j][p];
        }
        cumsum += p_sum[num_threads][p];
      }

      CHECK_EQ(cumsum, num_edges);

      for (int p = 0; p < num_partitions; p++) {
        src_gnids_list[p] = IdArray::Empty({p_sum[num_threads][p]}, src_gnids->dtype, src_gnids->ctx);
        dst_gnids_list[p] = IdArray::Empty({p_sum[num_threads][p]}, src_gnids->dtype, src_gnids->ctx);

        src_gnids_ptr_list[p] = src_gnids_list[p].Ptr<int64_t>();
        dst_gnids_ptr_list[p] = dst_gnids_list[p].Ptr<int64_t>();
      }
    }
#pragma omp barrier

    std::vector<int64_t> data_pos(p_sum[thread_id]);

    for (int64_t i = start_i; i < end_i; i++) {
      auto src_gnid = src_gnids_data[i];
      auto dst_gnid = dst_gnids_data[i];
      int part_id = part_data[i];
      int64_t offset = data_pos[part_id]++;
      *(src_gnids_ptr_list[part_id] + offset) = src_gnid;
      *(dst_gnids_ptr_list[part_id] + offset) = dst_gnid;
    }
  }

  return std::make_pair(src_gnids_list, dst_gnids_list);

  // Single cpu implementation.
  //
  // std::vector<std::vector<int64_t>> src_gnids_vectors(num_partitions);
  // std::vector<std::vector<int64_t>> dst_gnids_vectors(num_partitions);

  // for (int i = 0; i < num_partitions; i++) {
  //   src_gnids_vectors[i].reserve((int64_t)(1.2 * num_edges / num_partitions));
  //   dst_gnids_vectors[i].reserve((int64_t)(1.2 * num_edges / num_partitions));
  // }

  // for (int i = 0; i < num_edges; i++) {
  //   int part_id = -1;
  //   auto src_gnid = src_gnids_data[i];
  //   auto dst_gnid = dst_gnids_data[i];

  //   for (int j = 0; j < num_machines; j++) {
  //     if (src_gnid < nid_partitions_data[j + 1]) {
  //       part_id = j * num_gpus_per_machine + src_gnid % num_gpus_per_machine;
  //       break;
  //     }
  //   }

  //   if (part_id == -1) {
  //     part_id = target_id_to_part_id[src_gnid];
  //   }

  //   src_gnids_vectors[part_id].push_back(src_gnid);
  //   dst_gnids_vectors[part_id].push_back(dst_gnid);
  // }

  // std::vector<IdArray> src_gnids_list(num_partitions);
  // std::vector<IdArray> dst_gnids_list(num_partitions);

  // // runtime::parallel_for(0, num_partitions, [&](size_t b, size_t e) {
  // //   for (int i = b; i < e; i++) {
  // //     src_gnids_list[i] = NDArray::FromVector(src_gnids_vectors[i]);
  // //     dst_gnids_list[i] = NDArray::FromVector(dst_gnids_vectors[i]);
  // //   }
  // // });
  // for (int i = 0; i < num_partitions; i++) {
  //   src_gnids_list[i] = NDArray::FromVector(src_gnids_vectors[i]);
  //   dst_gnids_list[i] = NDArray::FromVector(dst_gnids_vectors[i]);
  // }

  // return std::make_pair(src_gnids_list, dst_gnids_list);
}

}
}
