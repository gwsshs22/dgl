/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/sampler_v2.h
 * \brief 
 */
#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>

#include <dgl/base_heterograph.h>
#include <dgl/runtime/object.h>
#include <dgl/runtime/packed_func.h>

namespace dgl {
namespace omega {

using namespace dgl::runtime;

class SamplingExecutorV2 : public Object {

 public:
  typedef std::pair<std::vector<std::pair<HeteroGraphPtr, IdArray>>, std::vector<NDArray>> sampling_ret;

  SamplingExecutorV2(
    const int num_machines,
    const int machine_rank,
    const int num_gpus_per_machine_in_group,
    const int gpu_rank_in_group,
    const int local_rank,
    const std::vector<int64_t>& nid_partitions,
    const int num_layers,
    const std::vector<int64_t>& fanouts,
    const bool is_cgp,
    const int recom_threshold,
    const PackedFunc& pull_fn,
    const PackedFunc& dist_sampling_fn,
    const PackedFunc& pe_recom_policy_fn,
    const PackedFunc& all_gather_fn,
    const PackedFunc& dist_edges_fn,
    const HeteroGraphRef& empty_graph_ref,
    const HeteroGraphRef& local_graph_ref,
    const IdArray& local_graph_global_id_mapping,
    const std::unordered_map<std::string, NDArray>& local_data_store,
    const IdArray& in_degrees,
    const IdArray& out_degrees,
    const IdArray& gnid_to_local_id_mapping);

  sampling_ret SampleBlocksDp(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids);

  sampling_ret SampleBlocksPrecoms(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const IdArray& new_lhs_ids_prefix);

  std::pair<sampling_ret, IdArray> SampleBlocksDpPrecomsWithRecom(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids);

  std::tuple<sampling_ret, IdArray, IdArray> SampleBlocksCgpWithRecom(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const IdArray& new_lhs_ids_prefix);

  static constexpr const char* _type_key = "omega.sampler_v2.SamplingExecutorV2";
  DGL_DECLARE_OBJECT_TYPE_INFO(SamplingExecutorV2, runtime::Object);

 private:
  std::pair<HeteroGraphPtr, IdArray> ToBlock_(
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const IdArray& new_lhs_ids_prefix) const;

  std::pair<IdArray, IdArray> DistributeEdges(
    const IdArray& target_gnids,
    const std::vector<int64_t>& num_assigned_targets,
    const IdArray& u,
    const IdArray& v);

  std::pair<IdArray, IdArray> LocalInEdges(
    const std::vector<int64_t>& num_assigned_targets,
    const IdArray& recom_block_target_ids,
    const IdArray& recom_block_num_assigned_target_nodes);

  std::pair<std::vector<IdArray>, std::vector<IdArray>> DistSampling(
    const IdArray& seeds, const int fanout) const;

  PackedFunc SendRemoteDistSamplingReqs(
    const int fanout,
    const std::vector<int>& part_ids,
    const std::vector<IdArray>& local_nids_list) const;

  std::pair<IdArray, IdArray> LocalSampling(const IdArray& seeds, const int fanout) const;

  NDArray Pull(const std::string& name, const IdArray& id_arr) const;

  PackedFunc SendRemotePullReqs(
    const std::string& name,
    const std::vector<int>& part_ids,
    const std::vector<IdArray>& local_nids_list) const;

  void CopyLocalData(
    NDArray& ret_tensor,
    const NDArray& local_tensor,
    const IdArray& local_nids,
    const IdArray& orig_pos,
    const int row_size) const;

  void CopyFetchedData(
    NDArray& ret_tensor,
    const NDArray& fetched_tensor,
    const IdArray& orig_pos,
    const int row_size) const;

  const int num_machines_;
  const int machine_rank_;
  const int num_gpus_per_machine_in_group_;
  const int gpu_rank_in_group_;
  const int local_rank_;
  const std::vector<int64_t> nid_partitions_;
  const int num_layers_;
  const std::vector<int64_t> fanouts_;
  const bool full_sampling_;
  const bool is_cgp_;
  const int recom_threshold_;
  const PackedFunc pull_fn_;
  const PackedFunc dist_sampling_fn_;
  const PackedFunc pe_recom_policy_fn_;
  const PackedFunc all_gather_fn_;
  const PackedFunc dist_edges_fn_;
  const HeteroGraphRef empty_graph_ref_;
  const HeteroGraphRef local_graph_ref_;
  const IdArray local_graph_global_id_mapping_;
  const std::unordered_map<std::string, NDArray> local_data_store_;
  const IdArray in_degrees_;
  const IdArray out_degrees_;
  const IdArray gnid_to_local_id_mapping_;
  const IdArray null_array_;
  const DGLContext cpu_ctx_;
  const DGLContext gpu_ctx_;
};

DGL_DEFINE_OBJECT_REF(SamplingExecutorV2Ref, SamplingExecutorV2);

}
}