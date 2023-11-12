#include "sampler_v2.h"

#include <algorithm>
#include <chrono>

#include <dgl/runtime/container.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/device_api.h>
#include <dgl/sampling/neighbor.h>
#include <dgl/aten/array_ops.h>

#include "../c_api_common.h"
#include "distributed_block.h"
#include "trace.h"
#include "sampling.h"

namespace dgl {
namespace omega {

std::pair<HeteroGraphPtr, IdArray> ToBlockGPU(const IdArray& u,
                                              const IdArray& v,
                                              const IdArray& dst_ids,
                                              const IdArray& src_ids,
                                              const IdArray& new_lhs_ids_prefix);

using namespace dgl::runtime;
using namespace dgl::aten;

namespace {

bool _IsFullSampling(const std::vector<int64_t>& fanouts) {
  bool is_full_sampling = fanouts[0] == -1;
  if (is_full_sampling) {
    for (auto f : fanouts)
      CHECK_EQ(f, -1);
  }

  return is_full_sampling;
}

inline IdArray View(IdArray& arr, int64_t offset) {
  return arr.CreateView({ arr->shape[0] - offset }, arr->dtype, offset * sizeof(int64_t));
}

std::vector<int64_t> GetNumAssignedTargetsPerGpu_(
    int num_machines, int num_gpus_per_machine, int num_targets) {
  int num_total_gpus = num_machines * num_gpus_per_machine;
  std::vector<int64_t> num_targets_per_gpu(num_total_gpus);

  for (int machine_idx = 0; machine_idx < num_machines; machine_idx++) {
    int64_t num_targets_in_machine;
    if (machine_idx < num_targets % num_machines) {
      num_targets_in_machine = num_targets / num_machines + 1;
    } else {
      num_targets_in_machine = num_targets / num_machines;
    }

    for (int gpu_idx = 0; gpu_idx < num_gpus_per_machine; gpu_idx++) {
      int global_gpu_idx = machine_idx * num_gpus_per_machine + gpu_idx;
      if (gpu_idx < num_targets_in_machine % num_gpus_per_machine) {
        num_targets_per_gpu[global_gpu_idx] = num_targets_in_machine / num_gpus_per_machine + 1;
      } else {
        num_targets_per_gpu[global_gpu_idx] = num_targets_in_machine / num_gpus_per_machine;
      }
    }
  }

  return num_targets_per_gpu;
}

}

std::pair<HeteroGraphPtr, IdArray> SamplingExecutorV2::ToBlock_(
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const IdArray& new_lhs_ids_prefix) const {
  CHECK_EQ(target_gnids->ctx.device_type, kDGLCUDA);
  CHECK_EQ(src_gnids->ctx.device_type, kDGLCUDA);
  CHECK_EQ(dst_gnids->ctx.device_type, kDGLCUDA);
  CHECK_EQ(new_lhs_ids_prefix->ctx.device_type, kDGLCUDA);
  return ToBlockGPU(
    src_gnids,
    dst_gnids,
    target_gnids,
    null_array_,
    new_lhs_ids_prefix);
}

SamplingExecutorV2::sampling_ret SamplingExecutorV2::SampleBlocksDp(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids) {

  std::vector<std::pair<HeteroGraphPtr, IdArray>> blocks;
  std::vector<NDArray> src_inputs_list;

  IdArray root_srcs, root_dsts;
  const int64_t num_targets = target_gnids->shape[0];

  std::vector<std::pair<IdArray, IdArray>> sample_ret;
  if (full_sampling_) {
    root_srcs = src_gnids;
    root_dsts = dst_gnids;
  } else {
    sample_ret = SampleEdgesGPU(
      target_gnids,
      src_gnids,
      dst_gnids,
      fanouts_);
    root_srcs = sample_ret[0].first;
    root_dsts = sample_ret[0].second;
  }

  auto root_block_ret = ToBlock_(
    target_gnids,
    root_srcs,
    root_dsts,
    target_gnids);

  blocks.insert(blocks.begin(), root_block_ret);

  for (int layer_idx = 1; layer_idx < num_layers_; layer_idx++) {
    auto src_ids = blocks[0].second;
    auto seeds = src_ids.CreateView({ src_ids->shape[0] - num_targets }, src_ids->dtype, num_targets * sizeof(int64_t));
    std::pair<std::vector<IdArray>, std::vector<IdArray>> dist_sampling_ret;
    {
      auto start_time = std::chrono::high_resolution_clock::now();
      dist_sampling_ret = DistSampling(seeds, fanouts_[layer_idx]);
      auto end_time = std::chrono::high_resolution_clock::now();
      PutTrace(batch_id, "fetch_edges", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
      int64_t fetch_size = 0;
      for (const auto& a : dist_sampling_ret.first) {
        fetch_size += a.GetSize();
      }
      for (const auto& a : dist_sampling_ret.second) {
        fetch_size += a.GetSize();
      }
      PutTrace(batch_id, "fetch_size_edges", fetch_size);
    }

    auto src_list = dist_sampling_ret.first;
    auto dst_list = dist_sampling_ret.second;

    if (full_sampling_) {
      src_list.insert(src_list.begin(), root_srcs);
      dst_list.insert(dst_list.begin(), root_dsts);
      src_list.push_back(root_dsts);
      dst_list.push_back(root_srcs);
    } else {
      src_list.insert(src_list.begin(), sample_ret[layer_idx].first);
      dst_list.insert(dst_list.begin(), sample_ret[layer_idx].second);
    }

    auto all_srcs = dgl::aten::Concat(src_list);
    auto all_dsts = dgl::aten::Concat(dst_list);

    auto block_ret = ToBlock_(
      src_ids,
      all_srcs,
      all_dsts,
      target_gnids);

    blocks.insert(blocks.begin(), block_ret);
  }

  auto src_ids = blocks[0].second;
  auto fetch_nids = src_ids.CreateView({ src_ids->shape[0] - num_targets }, src_ids->dtype, num_targets * sizeof(int64_t));
  NDArray input_features;
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    input_features = Pull("features", fetch_nids);
    auto end_time = std::chrono::high_resolution_clock::now();

    PutTrace(batch_id, "fetch_features", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    PutTrace(batch_id, "fetch_size_features", input_features.GetSize());
  }

  src_inputs_list.push_back(input_features);
  for (int i = 0; i < num_layers_ - 1; i++) {
    src_inputs_list.push_back(null_array_);
  }

  return std::make_pair(std::move(blocks), std::move(src_inputs_list));
}

SamplingExecutorV2::sampling_ret SamplingExecutorV2::SampleBlocksPrecoms(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const IdArray& new_lhs_ids_prefix) {

  std::vector<std::pair<HeteroGraphPtr, IdArray>> blocks;
  std::vector<NDArray> src_inputs_list;

  if (full_sampling_) {
    auto ret = ToBlock_(
        target_gnids,
        src_gnids,
        dst_gnids,
        new_lhs_ids_prefix);

    for (int i = 0; i < num_layers_; i++) {
      blocks.insert(blocks.begin(), ret);
    }
  } else {
    auto sampled_edges = SampleEdgesGPU(target_gnids, src_gnids, dst_gnids, fanouts_);

    for (int i = 0; i < num_layers_; i++) {
      auto ret = ToBlock_(
        target_gnids,
        sampled_edges[i].first,
        sampled_edges[i].second,
        new_lhs_ids_prefix
      );

      blocks.insert(blocks.begin(), ret);
    }
  }

  for (int layer_idx = 0; layer_idx < num_layers_; layer_idx++) {
    std::string row_name;
    if (layer_idx == 0) {
      row_name = "features";
    } else {
      row_name = "layer_" + std::to_string(layer_idx - 1);
    }

    int64_t num_src_nodes = blocks[layer_idx].first->NumVertices(0);
    int64_t num_dst_nodes = new_lhs_ids_prefix->shape[0];

    IdArray src_node_ids = blocks[layer_idx].second.CreateView(
      { num_src_nodes - num_dst_nodes },
      blocks[layer_idx].second->dtype,
      num_dst_nodes * sizeof(int64_t));

    NDArray src_inputs;
    {
      auto start_time = std::chrono::high_resolution_clock::now();
      src_inputs = Pull(row_name, src_node_ids);
      auto end_time = std::chrono::high_resolution_clock::now();
      PutTrace(batch_id, "fetch_" + row_name, std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
      PutTrace(batch_id, "fetch_size_" + row_name, src_inputs.GetSize());
    }
    src_inputs_list.push_back(src_inputs);
  }

  return std::make_pair(blocks, src_inputs_list);
}

std::pair<SamplingExecutorV2::sampling_ret, IdArray> SamplingExecutorV2::SampleBlocksDpPrecomsWithRecom(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids) {
  std::vector<std::pair<HeteroGraphPtr, IdArray>> blocks;
  std::vector<NDArray> src_inputs_list;
  const int64_t batch_size = target_gnids->shape[0];

  auto block = ToBlock_(
      target_gnids,
      src_gnids,
      dst_gnids,
      target_gnids);
  blocks.insert(blocks.begin(), block);

  auto reversed_block = ToBlock_(
      blocks[0].second,
      dst_gnids,
      src_gnids,
      blocks[0].second);

  auto direct_neighbor_ids = View(blocks[0].second, batch_size);
  auto direct_neighbor_ids_cpu = direct_neighbor_ids.CopyTo(cpu_ctx_);

  auto new_degrees = reversed_block.first->InDegrees(0, Range(batch_size, blocks[0].second->shape[0], blocks[0].second->dtype.bits, gpu_ctx_)).CopyTo(cpu_ctx_);

  CHECK_EQ(direct_neighbor_ids_cpu->shape[0], new_degrees->shape[0]);


  IdArray recompute_ids;
  IdArray reuse_ids;
  IdArray recompute_mask;
  IdArray recompute_pos;

  {
    auto start_time = std::chrono::high_resolution_clock::now();
    List<Value> policy_fn_inputs;
    policy_fn_inputs.push_back(Value(MakeValue(direct_neighbor_ids_cpu)));
    policy_fn_inputs.push_back(Value(MakeValue(new_degrees)));
    List<Value> recom_policy_ret = pe_recom_policy_fn_(policy_fn_inputs);

    recompute_ids = recom_policy_ret[0]->data;
    reuse_ids = recom_policy_ret[1]->data;
    recompute_mask = recom_policy_ret[2]->data;
    recompute_pos = recom_policy_ret[3]->data;
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "recom_policy", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
  }

  std::pair<std::vector<IdArray>, std::vector<IdArray>> dist_sampling_ret;
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    dist_sampling_ret = DistSampling(recompute_ids, -1);
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_edges", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    int64_t fetch_size = 0;
    for (const auto& a : dist_sampling_ret.first) {
      fetch_size += a.GetSize();
    }
    for (const auto& a : dist_sampling_ret.second) {
      fetch_size += a.GetSize();
    }
    PutTrace(batch_id, "fetch_size_edges", fetch_size);
  }

  auto recom_block_src_list = dist_sampling_ret.first;
  auto recom_block_dst_list = dist_sampling_ret.second;

  auto edge_arr = reversed_block.first->InEdges(0, Add(recompute_pos, batch_size).CopyTo(gpu_ctx_));
  recom_block_src_list.insert(recom_block_src_list.begin(), IndexSelect(reversed_block.second, edge_arr.src));
  recom_block_dst_list.insert(recom_block_dst_list.begin(), IndexSelect(reversed_block.second, edge_arr.dst));

  recom_block_src_list.insert(recom_block_src_list.begin(), src_gnids);
  recom_block_dst_list.insert(recom_block_dst_list.begin(), dst_gnids);

  auto recom_block_target_ids = Concat({ target_gnids, recompute_ids.CopyTo(gpu_ctx_) });
  auto recom_block = ToBlock_(
    recom_block_target_ids,
    Concat(recom_block_src_list),
    Concat(recom_block_dst_list),
    recom_block_target_ids);

  blocks.insert(blocks.begin(), recom_block);

  {
    auto start_time = std::chrono::high_resolution_clock::now();
    NDArray src_inputs = Pull("features", View(recom_block.second, batch_size));
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_features", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    PutTrace(batch_id, "fetch_size_features", src_inputs.GetSize());
    src_inputs_list.push_back(src_inputs);
  }
  
  for (int layer_idx = 1; layer_idx < num_layers_ - 1; layer_idx++) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto name = "layer_" + std::to_string(layer_idx - 1);
    NDArray src_inputs = Pull(name, View(recom_block.second, batch_size + recompute_ids->shape[0]));
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_" + name, std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    PutTrace(batch_id, "fetch_size_" + name, src_inputs.GetSize());
    src_inputs_list.push_back(src_inputs);
  }

  {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto name = "layer_" + std::to_string(num_layers_ - 2);
    NDArray src_inputs = Pull(name, reuse_ids);
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_" + name, std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    PutTrace(batch_id, "fetch_size_" + name, src_inputs.GetSize());
    src_inputs_list.push_back(src_inputs);
  }

  return std::make_pair(std::make_pair(blocks, src_inputs_list), recompute_mask);
}

std::tuple<SamplingExecutorV2::sampling_ret, IdArray, IdArray> SamplingExecutorV2::SampleBlocksCgpWithRecom(
    const int batch_id,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const IdArray& new_lhs_ids_prefix) {

  std::vector<std::pair<HeteroGraphPtr, IdArray>> blocks;
  std::vector<NDArray> src_inputs_list;
  auto num_assigned_targets = GetNumAssignedTargetsPerGpu_(num_machines_, num_gpus_per_machine_in_group_, target_gnids->shape[0]);
  const int64_t num_local_target_nodes = new_lhs_ids_prefix->shape[0];
  CHECK_EQ(num_assigned_targets[gpu_rank_in_group_], num_local_target_nodes);
  auto block = ToBlock_(
      target_gnids,
      src_gnids,
      dst_gnids,
      new_lhs_ids_prefix);
  blocks.insert(blocks.begin(), block);

  auto reversed_block = ToBlock_(
      blocks[0].second,
      dst_gnids,
      src_gnids,
      blocks[0].second);

  auto direct_neighbor_ids = View(blocks[0].second, num_local_target_nodes);
  auto direct_neighbor_ids_cpu = direct_neighbor_ids.CopyTo(cpu_ctx_);

  auto new_degrees = reversed_block.first->InDegrees(0, Range(num_local_target_nodes, blocks[0].second->shape[0], blocks[0].second->dtype.bits, gpu_ctx_)).CopyTo(cpu_ctx_);

  CHECK_EQ(direct_neighbor_ids_cpu->shape[0], new_degrees->shape[0]);

  IdArray recompute_ids;
  IdArray reuse_ids;
  IdArray recompute_mask;
  IdArray recompute_pos;
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    List<Value> policy_fn_inputs;
    policy_fn_inputs.push_back(Value(MakeValue(direct_neighbor_ids_cpu)));
    policy_fn_inputs.push_back(Value(MakeValue(new_degrees)));
    List<Value> recom_policy_ret = pe_recom_policy_fn_(policy_fn_inputs);

    recompute_ids = recom_policy_ret[0]->data;
    reuse_ids = recom_policy_ret[1]->data;
    recompute_mask = recom_policy_ret[2]->data;
    recompute_pos = recom_policy_ret[3]->data;
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "recom_policy", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
  }

  auto recom_block_local_target_ids = Concat({ new_lhs_ids_prefix.CopyTo(cpu_ctx_), recompute_ids });

  List<Value> all_gather_fn_inputs;
  all_gather_fn_inputs.push_back(Value(MakeValue(recom_block_local_target_ids)));
  List<Value> all_gather_ret = all_gather_fn_(all_gather_fn_inputs);
  IdArray recom_block_target_ids = all_gather_ret[0]->data;
  IdArray recom_block_num_assigned_target_nodes = all_gather_ret[1]->data;

  auto edge_arr = reversed_block.first->InEdges(0, Add(recompute_pos, num_local_target_nodes).CopyTo(gpu_ctx_));

  std::pair<IdArray, IdArray> dist_edges_ret;
  {
    auto global_src_ids = IndexSelect(reversed_block.second, edge_arr.src);
    auto global_dst_ids = IndexSelect(reversed_block.second, edge_arr.dst);
    auto start_time = std::chrono::high_resolution_clock::now();
    dist_edges_ret = DistributeEdges(
      target_gnids,
      num_assigned_targets,
      global_src_ids,
      global_dst_ids
    );
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_edges", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    int64_t fetch_size = dist_edges_ret.first.GetSize() + dist_edges_ret.second.GetSize();
    PutTrace(batch_id, "fetch_size_edges", fetch_size);
  }

  std::vector<IdArray> recom_src_ids_list, recom_dst_ids_list;
  recom_src_ids_list.push_back(dist_edges_ret.first);
  recom_dst_ids_list.push_back(dist_edges_ret.second);

  auto local_in_edges_ret = LocalInEdges(
    num_assigned_targets,
    recom_block_target_ids,
    recom_block_num_assigned_target_nodes);
  
  recom_src_ids_list.insert(recom_src_ids_list.begin(), local_in_edges_ret.first);
  recom_dst_ids_list.insert(recom_dst_ids_list.begin(), local_in_edges_ret.second);

  recom_src_ids_list.insert(recom_src_ids_list.begin(), src_gnids);
  recom_dst_ids_list.insert(recom_dst_ids_list.begin(), dst_gnids);

  auto recom_block = ToBlock_(
    recom_block_target_ids.CopyTo(gpu_ctx_),
    Concat(recom_src_ids_list),
    Concat(recom_dst_ids_list),
    recom_block_local_target_ids.CopyTo(gpu_ctx_));

  blocks.insert(blocks.begin(), recom_block);

  {
    auto start_time = std::chrono::high_resolution_clock::now();
    NDArray src_inputs = Pull("features", View(recom_block.second, num_local_target_nodes));
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_features", std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    PutTrace(batch_id, "fetch_size_features", src_inputs.GetSize());
    src_inputs_list.push_back(src_inputs);
  }
  
  for (int layer_idx = 1; layer_idx < num_layers_ - 1; layer_idx++) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto name = "layer_" + std::to_string(layer_idx - 1);
    NDArray src_inputs = Pull(name, View(recom_block.second, num_local_target_nodes + recompute_ids->shape[0]));
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_" + name, std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    PutTrace(batch_id, "fetch_size_" + name, src_inputs.GetSize());
    src_inputs_list.push_back(src_inputs);
  }

  {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto name = "layer_" + std::to_string(num_layers_ - 2);
    NDArray src_inputs = Pull(name, reuse_ids);
    auto end_time = std::chrono::high_resolution_clock::now();
    PutTrace(batch_id, "fetch_" + name, std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    PutTrace(batch_id, "fetch_size_" + name, src_inputs.GetSize());
    src_inputs_list.push_back(src_inputs);
  }
  
  return std::make_tuple(std::make_pair(blocks, src_inputs_list), recompute_mask, recom_block_num_assigned_target_nodes);
}

std::pair<IdArray, IdArray> SamplingExecutorV2::DistributeEdges(
  const IdArray& target_gnids,
  const std::vector<int64_t>& num_assigned_targets,
  const IdArray& u,
  const IdArray& v) {

  auto split_ret = SplitEdgesGPU(target_gnids, num_assigned_targets, u, v);

  List<Value> src_ids_list;
  List<Value> dst_ids_list;

  for (int i = 0; i < split_ret.first.size(); i++) {
    src_ids_list.push_back(Value(MakeValue(split_ret.first[i])));
    dst_ids_list.push_back(Value(MakeValue(split_ret.second[i])));
  }

  List<Value> dist_edges_ret = dist_edges_fn_(src_ids_list, dst_ids_list);
  IdArray shuffled_src_ids = dist_edges_ret[0]->data;
  IdArray shuffled_dst_ids = dist_edges_ret[1]->data;

  return std::make_pair(shuffled_src_ids, shuffled_dst_ids);
}

std::pair<IdArray, IdArray> SamplingExecutorV2::LocalInEdges(
  const std::vector<int64_t>& num_assigned_targets,
  const IdArray& recom_block_target_ids,
  const IdArray& recom_block_num_assigned_target_nodes) {

  auto device = dgl::runtime::DeviceAPI::Get(cpu_ctx_);
  const int num_parts = num_assigned_targets.size();
  int64_t total_nodes = 0;

  for (int i = 0; i < num_parts; i++) {
    total_nodes += *(recom_block_num_assigned_target_nodes.Ptr<int64_t>() + i) - num_assigned_targets[i];
  }

  int64_t from_offset = 0;
  int64_t to_offset = 0;
  IdArray nids = NewIdArray(total_nodes);
  for (int i = 0; i < num_parts; i++) {
    int64_t num_copy = *(recom_block_num_assigned_target_nodes.Ptr<int64_t>() + i) - num_assigned_targets[i];
    device->CopyDataFromTo(
      recom_block_target_ids->data, (from_offset + num_assigned_targets[i]) * sizeof(int64_t),
      nids->data, to_offset * sizeof(int64_t), num_copy * sizeof(int64_t), cpu_ctx_, cpu_ctx_, nids->dtype);

    from_offset += *(recom_block_num_assigned_target_nodes.Ptr<int64_t>() + i);
    to_offset += num_copy;
  }

  CHECK_EQ(to_offset, total_nodes);

  std::vector<int64_t> filtered_nids;
  filtered_nids.reserve(total_nodes);

  auto nids_ptr = nids.Ptr<int64_t>();
  auto ptr = gnid_to_local_id_mapping_.Ptr<int64_t>();

  for (int i = 0; i < total_nodes; i++) {
    int64_t mapped_id = *(ptr + nids_ptr[i]);
    if (mapped_id >= 0) {
      filtered_nids.push_back(mapped_id);
    }
  }

  nids = VecToIdArray(filtered_nids);
  auto edge_arr = local_graph_ref_->InEdges(0, nids);
  return std::make_pair(
    IndexSelect(local_graph_global_id_mapping_, edge_arr.src).CopyTo(gpu_ctx_),
    IndexSelect(local_graph_global_id_mapping_, edge_arr.dst).CopyTo(gpu_ctx_));
}

std::pair<std::vector<IdArray>, std::vector<IdArray>> SamplingExecutorV2::DistSampling(
    const IdArray& seeds_gpu, const int fanout) const {
  IdArray seeds;
  if (seeds_gpu->ctx.device_type != kDGLCPU) {
    seeds = seeds_gpu.CopyTo(cpu_ctx_);
  } else {
    seeds = seeds_gpu;
  }

  const int64_t num_seeds = seeds->shape[0];
  const int64_t* seeds_ptr = seeds.Ptr<int64_t>();
  const int num_partitions = nid_partitions_.size() - 1;

  std::vector<IdArray> local_nids_list(num_partitions);
  std::vector<int64_t*> local_nids_ptr_list(num_partitions, nullptr);

  const int num_threads = runtime::compute_num_threads(0, num_seeds, 1);
  std::vector<std::vector<int64_t>> p_sum;
  p_sum.resize(num_threads + 1);
  p_sum[0].resize(num_partitions, 0);
  std::unique_ptr<int[]> part_data(new int[num_seeds]);
#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();
    const int64_t start_i =
        thread_id * (num_seeds / num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_seeds % num_threads);
    const int64_t end_i =
        (thread_id + 1) * (num_seeds / num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_seeds % num_threads);
    assert(thread_id + 1 < num_threads || end_i == num_seeds);

    p_sum[thread_id + 1].resize(num_partitions, 0);

    for (int64_t i = start_i; i < end_i; i++) {
      int part_id = -1;
      auto nid = seeds_ptr[i];

      for (int j = 0; j < num_partitions; j++) {
        if (nid < nid_partitions_[j + 1]) {
          part_id = j;
          break;
        }
      }

      CHECK_GE(part_id, 0);
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

      CHECK_EQ(cumsum, num_seeds);

      for (int p = 0; p < num_partitions; p++) {
        const int64_t num_nodes_in_part = p_sum[num_threads][p];

        local_nids_list[p] = IdArray::Empty({ num_nodes_in_part }, seeds->dtype, seeds->ctx);
        local_nids_ptr_list[p] = local_nids_list[p].Ptr<int64_t>();
      }
    }
#pragma omp barrier

    std::vector<int64_t> data_pos(p_sum[thread_id]);

    for (int64_t i = start_i; i < end_i; i++) {
      int64_t nid = seeds_ptr[i];
      int part_id = part_data[i];

      int64_t offset = data_pos[part_id]++;
      *(local_nids_ptr_list[part_id] + offset) = nid - nid_partitions_[part_id];
    }
  }

  std::vector<int> dist_sampling_part_ids;
  std::vector<IdArray> dist_sampling_local_ids;

  for (int i = 0; i < num_partitions; i++) {
    if (i == machine_rank_ || local_nids_list[i]->shape[0] == 0) continue;
    dist_sampling_part_ids.push_back(i);
    dist_sampling_local_ids.push_back(local_nids_list[i]);
  }

  const int num_reqs = dist_sampling_part_ids.size();
  auto recv_fn = SendRemoteDistSamplingReqs(fanout, dist_sampling_part_ids, dist_sampling_local_ids);
  auto local_ret = LocalSampling(local_nids_list[machine_rank_], fanout);
  std::vector<IdArray> sampled_srcs_list, sampled_dsts_list;
  sampled_srcs_list.push_back(local_ret.first);
  sampled_dsts_list.push_back(local_ret.second);

  if (num_reqs > 0) {
    List<Value> ret = recv_fn();
    CHECK_EQ(ret.size(), num_reqs * 2);
    for (int i = 0; i < num_reqs; i++) {
      IdArray srcs = ret[2 * i]->data;
      IdArray dsts = ret[2 * i + 1]->data;
      sampled_srcs_list.push_back(srcs.CopyTo(gpu_ctx_));
      sampled_dsts_list.push_back(dsts.CopyTo(gpu_ctx_));
    }
  }

  return { sampled_srcs_list, sampled_dsts_list };
}


PackedFunc SamplingExecutorV2::SendRemoteDistSamplingReqs(
  const int fanout,
  const std::vector<int>& part_ids,
  const std::vector<IdArray>& local_nids_list) const {

  if (part_ids.size() == 0) {
    // Should not be called by callees.
    return PackedFunc();
  }

  List<Value> args_part_ids;
  List<Value> args_local_nids_list;

  for (int i = 0; i < part_ids.size(); i++) {
    args_part_ids.push_back(Value(MakeValue(part_ids[i])));
    args_local_nids_list.push_back(Value(MakeValue(local_nids_list[i])));
  }

  return dist_sampling_fn_(args_part_ids, args_local_nids_list, fanout);
}

std::pair<IdArray, IdArray> SamplingExecutorV2::LocalSampling(const IdArray& seeds, const int fanout) const {

  auto local_graph = local_graph_ref_.sptr();
  auto local_sampled_subg = dgl::sampling::SampleNeighbors(
    local_graph,
    { seeds },
    { fanout },
    dgl::EdgeDir::kIn,
    { null_array_ },
    { null_array_ },
    false);
  
  CHECK_EQ(local_sampled_subg.graph->NumEdgeTypes(), 1);
  auto local_edges = local_sampled_subg.graph->Edges(0);
  auto src_local_ids = local_edges.src;
  auto dst_local_ids = local_edges.dst;
  const int64_t num_edges = src_local_ids->shape[0];

  CHECK_EQ(dst_local_ids->shape[0], num_edges);
  const int64_t* src_local_ids_ptr = src_local_ids.Ptr<int64_t>();
  const int64_t* dst_local_ids_ptr = dst_local_ids.Ptr<int64_t>();
  IdArray src_global_ids = NDArray::Empty({ num_edges }, src_local_ids->dtype, src_local_ids->ctx);
  IdArray dst_global_ids = NDArray::Empty({ num_edges }, dst_local_ids->dtype, dst_local_ids->ctx);
  int64_t* src_global_ids_ptr = src_global_ids.Ptr<int64_t>();
  int64_t* dst_global_ids_ptr = dst_global_ids.Ptr<int64_t>();
  const int64_t* global_mapping_ptr = local_graph_global_id_mapping_.Ptr<int64_t>();

  parallel_for(0, num_edges, [&](size_t b, size_t e) {
    for (auto i = b; i < e; i++) {
      src_global_ids_ptr[i] = global_mapping_ptr[src_local_ids_ptr[i]];
      dst_global_ids_ptr[i] = global_mapping_ptr[dst_local_ids_ptr[i]];
    }
  });

  return { src_global_ids.CopyTo(gpu_ctx_), dst_global_ids.CopyTo(gpu_ctx_) };
}

NDArray SamplingExecutorV2::Pull(const std::string& name, const IdArray& id_arr_gpu) const {
  const IdArray& id_arr = id_arr_gpu.CopyTo(cpu_ctx_);
  const int64_t num_nodes = id_arr->shape[0];
  const int64_t* id_arr_ptr = id_arr.Ptr<int64_t>();
  const int num_partitions = nid_partitions_.size() - 1;

  std::vector<IdArray> orig_pos_list(num_partitions);
  std::vector<int64_t*> orig_pos_ptr_list(num_partitions, nullptr);

  std::vector<IdArray> local_nids_list(num_partitions);
  std::vector<int64_t*> local_nids_ptr_list(num_partitions, nullptr);

  const int num_threads = runtime::compute_num_threads(0, num_nodes, 1);
  std::vector<std::vector<int64_t>> p_sum;
  p_sum.resize(num_threads + 1);
  p_sum[0].resize(num_partitions, 0);
  std::unique_ptr<int[]> part_data(new int[num_nodes]);
#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();
    const int64_t start_i =
        thread_id * (num_nodes / num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_nodes % num_threads);
    const int64_t end_i =
        (thread_id + 1) * (num_nodes / num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_nodes % num_threads);
    assert(thread_id + 1 < num_threads || end_i == num_nodes);

    p_sum[thread_id + 1].resize(num_partitions, 0);

    for (int64_t i = start_i; i < end_i; i++) {
      int part_id = -1;
      auto nid = id_arr_ptr[i];

      for (int j = 0; j < num_partitions; j++) {
        if (nid < nid_partitions_[j + 1]) {
          part_id = j;
          break;
        }
      }

      CHECK_GE(part_id, 0);
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

      CHECK_EQ(cumsum, num_nodes);

      for (int p = 0; p < num_partitions; p++) {
        const int64_t num_nodes_in_part = p_sum[num_threads][p];

        orig_pos_list[p] = IdArray::Empty({ num_nodes_in_part }, id_arr->dtype, id_arr->ctx);
        orig_pos_ptr_list[p] = orig_pos_list[p].Ptr<int64_t>();

        local_nids_list[p] = IdArray::Empty({ num_nodes_in_part }, id_arr->dtype, id_arr->ctx);
        local_nids_ptr_list[p] = local_nids_list[p].Ptr<int64_t>();
      }
    }
#pragma omp barrier

    std::vector<int64_t> data_pos(p_sum[thread_id]);

    for (int64_t i = start_i; i < end_i; i++) {
      int64_t nid = id_arr_ptr[i];
      int part_id = part_data[i];

      int64_t offset = data_pos[part_id]++;
      *(orig_pos_ptr_list[part_id] + offset) = i;
      *(local_nids_ptr_list[part_id] + offset) = nid - nid_partitions_[part_id];
    }
  }

  std::vector<int> pull_part_ids;
  std::vector<IdArray> pull_local_ids;

  for (int i = 0; i < num_partitions; i++) {
    if (i == machine_rank_ || local_nids_list[i]->shape[0] == 0) continue;
    pull_part_ids.push_back(i);
    pull_local_ids.push_back(local_nids_list[i]);
  }

  auto recv_fn = SendRemotePullReqs(name, pull_part_ids, pull_local_ids);

  auto find_ret = local_data_store_.find(name);
  CHECK(find_ret != local_data_store_.end());
  NDArray local_tensor = find_ret->second;

  std::vector<int64_t> ret_tensor_shape;
  ret_tensor_shape.push_back(num_nodes);
  for (int i = 1; i < local_tensor->ndim; i++) {
    ret_tensor_shape.push_back(local_tensor->shape[i]);
  }

  NDArray ret_tensor = NDArray::Empty(ret_tensor_shape, local_tensor->dtype, local_tensor->ctx);
  int row_size = 1;
  for (int i = 1; i < ret_tensor->ndim; ++i) {
    row_size *= ret_tensor->shape[i];
  }
  row_size *= (ret_tensor->dtype.bits / 8);
  CHECK_GT(row_size, 0);

  CopyLocalData(ret_tensor, local_tensor, local_nids_list[machine_rank_], orig_pos_list[machine_rank_], row_size);

  const int num_remote_reqs = pull_part_ids.size();
  for (int i = 0; i < num_remote_reqs; i++) { 
    const List<Value> ret = recv_fn();
    const int part_id = ret[0]->data;
    const NDArray fetched_tensor = ret[1]->data;

    CopyFetchedData(ret_tensor, fetched_tensor, orig_pos_list[part_id], row_size);
  }

  return ret_tensor;
}

PackedFunc SamplingExecutorV2::SendRemotePullReqs(
    const std::string& name,
    const std::vector<int>& part_ids,
    const std::vector<IdArray>& local_nids_list) const {
  if (is_cgp_) {
    CHECK_EQ(part_ids.size(), 0);
  }

  if (part_ids.size() == 0) {
    // Should not be called by callees.
    return PackedFunc();
  }

  Value args_name = Value(MakeValue(name));
  List<Value> args_part_ids;
  List<Value> args_local_nids_list;

  for (int i = 0; i < part_ids.size(); i++) {
    args_part_ids.push_back(Value(MakeValue(part_ids[i])));
    args_local_nids_list.push_back(Value(MakeValue(local_nids_list[i])));
  }

  return pull_fn_(name, args_part_ids, args_local_nids_list);
}

void SamplingExecutorV2::CopyLocalData(
  NDArray& ret_tensor,
  const NDArray& local_tensor,
  const IdArray& local_nids,
  const IdArray& orig_pos,
  const int row_size) const {

  const int64_t num_local_nodes = local_nids->shape[0];
  if (num_local_nodes == 0) {
    return;
  }

  CHECK_GE(ret_tensor->shape[0], local_nids->shape[0]);
  CHECK_EQ(ret_tensor->ndim, local_tensor->ndim);
  for (int i = 1; i < ret_tensor->ndim; i++) {
    CHECK_EQ(ret_tensor->shape[i], local_tensor->shape[i]);
  }

  const int64_t* local_nids_ptr = local_nids.Ptr<int64_t>();
  const int64_t* orig_pos_ptr = orig_pos.Ptr<int64_t>();

  const char* local_data = local_tensor.Ptr<char>();
  char* return_data = ret_tensor.Ptr<char>();

  parallel_for(0, num_local_nodes, [&](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      memcpy(
          return_data + orig_pos_ptr[i] * row_size,
          local_data + local_nids_ptr[i] * row_size, row_size);
    }
  });
}

void SamplingExecutorV2::CopyFetchedData(
  NDArray& ret_tensor,
  const NDArray& fetched_tensor,
  const IdArray& orig_pos,
  const int row_size) const {

  const int64_t num_fetched_nodes = fetched_tensor->shape[0];
  CHECK_GT(num_fetched_nodes, 0);
  CHECK_GE(ret_tensor->shape[0], fetched_tensor->shape[0]);
  CHECK_EQ(ret_tensor->ndim, fetched_tensor->ndim);
  for (int i = 1; i < ret_tensor->ndim; i++) {
    CHECK_EQ(ret_tensor->shape[i], fetched_tensor->shape[i]);
  }

  const int64_t* orig_pos_ptr = orig_pos.Ptr<int64_t>();

  const char* fetched_data = fetched_tensor.Ptr<char>();
  char* return_data = ret_tensor.Ptr<char>();

  parallel_for(0, num_fetched_nodes, [&](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      memcpy(
          return_data + orig_pos_ptr[i] * row_size,
          fetched_data + i * row_size, row_size);
    }
  });
}

SamplingExecutorV2::SamplingExecutorV2(
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
    const IdArray& gnid_to_local_id_mapping) :
      num_machines_(num_machines),
      machine_rank_(machine_rank),
      num_gpus_per_machine_in_group_(num_gpus_per_machine_in_group),
      gpu_rank_in_group_(gpu_rank_in_group),
      local_rank_(local_rank),
      nid_partitions_(nid_partitions),
      num_layers_(num_layers),
      fanouts_(fanouts),
      recom_threshold_(recom_threshold),
      full_sampling_(_IsFullSampling(fanouts)),
      is_cgp_(is_cgp),
      pull_fn_(pull_fn),
      dist_sampling_fn_(dist_sampling_fn),
      pe_recom_policy_fn_(pe_recom_policy_fn),
      all_gather_fn_(all_gather_fn),
      dist_edges_fn_(dist_edges_fn),
      empty_graph_ref_(empty_graph_ref),
      local_graph_ref_(local_graph_ref),
      local_graph_global_id_mapping_(local_graph_global_id_mapping),
      local_data_store_(local_data_store),
      in_degrees_(in_degrees),
      out_degrees_(out_degrees),
      gnid_to_local_id_mapping_(gnid_to_local_id_mapping),
      null_array_(NDArray::FromVector(std::vector<int64_t>({}))),
      cpu_ctx_({ DGLDeviceType::kDGLCPU, 0 }),
      gpu_ctx_({ DGLDeviceType::kDGLCUDA, local_rank }) {

  CHECK_EQ(num_layers_, fanouts_.size());
}

}
}
