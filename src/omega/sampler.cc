#include "sampler.h"

#include <algorithm>

#include <dgl/runtime/container.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/sampling/neighbor.h>
#include <dgl/aten/array_ops.h>

#include "../c_api_common.h"

#include "distributed_block.h"
#include "sampling.h"

namespace dgl {
namespace omega {

using namespace dgl::runtime;

void SampleBlocksDp(
  const SamplingExecutor& executor,
  const SamplingExecutor::Task& task);

void SampleBlocksDpWithPrecoms(
  const SamplingExecutor& executor,
  const SamplingExecutor::Task& task);

void SampleBlocksCgp(
  const SamplingExecutor& executor,
  const SamplingExecutor::Task& task);

void DoneTask(
  const SamplingExecutor::Task& task,
  const std::vector<std::tuple<HeteroGraphPtr, IdArray, IdArray>>& blocks,
  const std::vector<NDArray>& src_inputs_list) {
  List<Value> ret_list;

  for (int i = 0; i < blocks.size(); i++) {
    ret_list.push_back(Value(MakeValue(HeteroGraphRef(std::get<0>(blocks[i])))));
    ret_list.push_back(Value(MakeValue(std::get<1>(blocks[i]))));
    ret_list.push_back(Value(MakeValue(std::get<2>(blocks[i]))));
    ret_list.push_back(Value(MakeValue(src_inputs_list[i])));
  }

  task.callback(ret_list);
}

void SampleBlocks(
  const SamplingExecutor& executor,
  const SamplingExecutor::Task& task) {

  if (executor.exec_mode() == "dp") {
    if (executor.use_precoms()) {
      SampleBlocksDpWithPrecoms(executor, task);
    } else {
      SampleBlocksDp(executor, task);
    }
  } else {
    SampleBlocksCgp(executor, task);
  }
}

void SampleBlocksDp(
  const SamplingExecutor& executor,
  const SamplingExecutor::Task& task) {

  std::vector<std::tuple<HeteroGraphPtr, IdArray, IdArray>> blocks;
  std::vector<NDArray> src_inputs_list;

  IdArray root_srcs, root_dsts;
  const int64_t num_targets = task.target_gnids->shape[0];

  if (executor.full_sampling()) {
    root_srcs = task.src_gnids;
    root_dsts = task.dst_gnids;
  } else {
    auto sample_ret = SampleEdges(
      task.target_gnids,
      task.src_gnids,
      task.dst_gnids,
      { executor.fanouts()[0] });
    root_srcs = sample_ret[0].first;
    root_dsts = sample_ret[0].second;
  }

  auto root_block_ret = executor.ToBlock_(
    task.target_gnids,
    root_srcs,
    root_dsts);
  blocks.insert(blocks.begin(), std::make_tuple(root_block_ret.first, root_block_ret.second, executor.null_array()));

  for (int layer_idx = 1; layer_idx < executor.num_layers(); layer_idx++) {
    auto src_ids = std::get<1>(blocks[0]);
    auto seeds = src_ids.CreateView({ src_ids->shape[0] - num_targets }, src_ids->dtype, num_targets * sizeof(int64_t));

    auto dist_sampling_ret = executor.DistSampling(seeds, executor.fanouts()[layer_idx]);
    auto src_list = dist_sampling_ret.first;
    auto dst_list = dist_sampling_ret.second;

    src_list.insert(src_list.begin(), root_srcs);
    dst_list.insert(dst_list.begin(), root_dsts);

    auto all_srcs = dgl::aten::Concat(src_list);
    auto all_dsts = dgl::aten::Concat(dst_list);

    auto block_ret = executor.ToBlock_(
      src_ids,
      all_srcs,
      all_dsts);

    blocks.insert(blocks.begin(), std::make_tuple(block_ret.first, block_ret.second, executor.null_array()));
  }

  auto src_ids = std::get<1>(blocks[0]);
  auto fetch_nids = src_ids.CreateView({ src_ids->shape[0] - num_targets }, src_ids->dtype, num_targets * sizeof(int64_t));
  NDArray input_features = executor.Pull("features", fetch_nids);

  src_inputs_list.push_back(input_features);
  for (int i = 0; i < executor.num_layers() - 1; i++) {
    src_inputs_list.push_back(executor.null_array());
  }

  DoneTask(task, blocks, src_inputs_list);
}

void SampleBlocksDpWithPrecoms(
  const SamplingExecutor& executor,
  const SamplingExecutor::Task& task) {

  std::vector<std::tuple<HeteroGraphPtr, IdArray, IdArray>> blocks;
  std::vector<NDArray> src_inputs_list;

  if (executor.full_sampling()) {
    auto ret = executor.ToBlock_(
        task.target_gnids,
        task.src_gnids,
        task.dst_gnids);
    
    for (int i = 0; i < executor.num_layers(); i++) {
      blocks.insert(blocks.begin(), std::make_tuple(ret.first, ret.second, executor.null_array()));
    }
  } else {
    auto sampled_edges = SampleEdges(task.target_gnids, task.src_gnids, task.dst_gnids, executor.fanouts());
    for (int i = 0; i < executor.num_layers(); i++) {
      auto ret = executor.ToBlock_(
        task.target_gnids,
        sampled_edges[i].first,
        sampled_edges[i].second
      );

      blocks.insert(blocks.begin(), std::make_tuple(ret.first, ret.second, executor.null_array()));
    }
  }

  for (int layer_idx = 0; layer_idx < executor.num_layers(); layer_idx++) {
    std::string row_name;
    if (layer_idx == 0) {
      row_name = "features";
    } else {
      row_name = "layer_" + std::to_string(layer_idx - 1);
    }


    auto graph_idx = std::get<0>(blocks[layer_idx]);
    auto block_src_ids = std::get<1>(blocks[layer_idx]);
    int64_t num_src_nodes = graph_idx->NumVertices(0);
    int64_t num_dst_nodes = graph_idx->NumVertices(1);

    IdArray src_node_ids = block_src_ids.CreateView(
      { num_src_nodes - num_dst_nodes },
      block_src_ids->dtype,
      num_dst_nodes * sizeof(int64_t));

    NDArray src_inputs = executor.Pull(row_name, src_node_ids);
    src_inputs_list.push_back(src_inputs);
  }

  DoneTask(task, blocks, src_inputs_list);
}

void SampleBlocksCgp(
  const SamplingExecutor& executor,
  const SamplingExecutor::Task& task) {
  std::vector<std::tuple<HeteroGraphPtr, IdArray, IdArray>> blocks;
  std::vector<NDArray> src_inputs_list;

  if (executor.full_sampling()) {
    auto ret = executor.ToDistributedBlock_(
        task.target_gnids,
        task.src_gnids,
        task.dst_gnids);
    
    for (int i = 0; i < executor.num_layers(); i++) {
      blocks.insert(blocks.begin(), ret);
    }
  } else {
    auto sampled_edges = SampleEdges(task.target_gnids, task.src_gnids, task.dst_gnids, executor.cgp_fanouts());
    for (int i = 0; i < executor.num_layers(); i++) {
      auto ret = executor.ToDistributedBlock_(
        task.target_gnids,
        sampled_edges[i].first,
        sampled_edges[i].second
      );

      blocks.insert(blocks.begin(), ret);
    }
  }
  
  const int64_t num_local_targets = executor.GetNumLocalTargets(task.target_gnids->shape[0]);
  for (int layer_idx = 0; layer_idx < executor.num_layers(); layer_idx++) {
    std::string row_name;
    if (layer_idx == 0) {
      row_name = "features";
    } else {
      row_name = "layer_" + std::to_string(layer_idx - 1);
    }

    auto graph_idx = std::get<0>(blocks[layer_idx]);
    auto block_src_ids = std::get<1>(blocks[layer_idx]);
    int64_t num_src_nodes = graph_idx->NumVertices(0);

    IdArray src_node_ids = block_src_ids.CreateView(
      { num_src_nodes - num_local_targets },
      block_src_ids->dtype,
      num_local_targets * sizeof(int64_t));

    NDArray src_inputs = executor.Pull(row_name, src_node_ids);
    src_inputs_list.push_back(src_inputs);
  }

  DoneTask(task, blocks, src_inputs_list);
}

static int _SamplingThreadMain(
  std::shared_ptr<SamplingExecutor::QueueT> queue,
  const SamplingExecutor* executor) {

  while (true) {
    std::shared_ptr<SamplingExecutor::Task> task;
    queue->wait_dequeue(task);

    if (task->batch_id < 0) {
      return 0;
    }

    SampleBlocks(*executor, *task);
  }

  return 0;
}

std::vector<int64_t> GetCgpFanouts(
  const int num_machines,
  const int machine_rank,
  const int num_gpus_per_machine_in_group,
  const int local_gpu_rank_in_group,
  const std::vector<int64_t>& fanouts) {

  std::vector<int64_t> cgp_fanouts(fanouts.size());
  if (fanouts[0] == -1) {
    for (int i = 0; i < fanouts.size(); i++) {
      cgp_fanouts[i] = -1;
    }
  } else {
    for (int i = 0; i < fanouts.size(); i++) {
      int fanout = fanouts[i];
      int idx = machine_rank * num_gpus_per_machine_in_group + local_gpu_rank_in_group;
      int cgp_fanout = GetNumAssignedTargetsPerGpu(num_machines, num_gpus_per_machine_in_group, fanout)[idx];
      cgp_fanout = std::max(cgp_fanout, 1);
      cgp_fanouts[i] = cgp_fanout;
    }
  }

  return cgp_fanouts;
}

bool _IsFullSampling(const std::vector<int64_t>& fanouts) {
  bool is_full_sampling = fanouts[0] == -1;
  if (is_full_sampling) {
    for (auto f : fanouts)
      CHECK_EQ(f, -1);
  }

  return is_full_sampling;
}

SamplingExecutor::SamplingExecutor(
    const int num_threads,
    const int num_machines,
    const int machine_rank,
    const int num_gpus_per_machine_in_group,
    const int local_gpu_rank_in_group,
    const std::vector<int64_t>& nid_partitions,
    const std::string& exec_mode,
    const bool use_precoms,
    const int num_layers,
    const std::vector<int64_t>& fanouts,
    const PackedFunc& pull_fn,
    const PackedFunc& dist_sampling_fn,
    const HeteroGraphRef& empty_graph_ref,
    const HeteroGraphRef& local_graph_ref,
    const IdArray& local_graph_global_id_mapping,
    const std::unordered_map<std::string, NDArray>& local_data_store) :
      queue_(new QueueT()),
      num_threads_(num_threads),
      num_machines_(num_machines),
      machine_rank_(machine_rank),
      num_gpus_per_machine_in_group_(num_gpus_per_machine_in_group),
      local_gpu_rank_in_group_(local_gpu_rank_in_group),
      nid_partitions_(nid_partitions),
      exec_mode_(exec_mode),
      use_precoms_(use_precoms),
      num_layers_(num_layers),
      fanouts_(fanouts),
      cgp_fanouts_(GetCgpFanouts(num_machines, machine_rank, num_gpus_per_machine_in_group, local_gpu_rank_in_group, fanouts)),
      full_sampling_(_IsFullSampling(fanouts)),
      pull_fn_(pull_fn),
      dist_sampling_fn_(dist_sampling_fn),
      empty_graph_ref_(empty_graph_ref),
      local_graph_ref_(local_graph_ref),
      local_graph_global_id_mapping_(local_graph_global_id_mapping),
      local_data_store_(local_data_store),
      null_array_(NDArray::FromVector(std::vector<int64_t>({}))) {

  CHECK_EQ(num_layers_, fanouts_.size());
  if (exec_mode_ == "cgp" || exec_mode_ == "cgp-multi") {
    CHECK(use_precoms_);
  }

  for (int i = 0; i < num_threads; i++) {
    thread_group_.create(
      "SamplingTaskExecutor[Thread-" + std::to_string(i) + "]",
      false,
      _SamplingThreadMain,
      queue_,
      this);
  }
}

void SamplingExecutor::Enqueue(std::shared_ptr<Task>&& task) {
  queue_->enqueue(std::move(task));
}

void SamplingExecutor::Shutdown() {
  const int shutdown_batch_id = -1;
  const IdArray dummy_arr;
  const PackedFunc dummy_callback;

  for (int i = 0; i < num_threads_; i++) {
    Enqueue(std::make_shared<Task>(
      shutdown_batch_id,
      dummy_arr,
      dummy_arr,
      dummy_arr,
      dummy_callback
    ));
  }

  thread_group_.request_shutdown_all();
  thread_group_.join_all();
}

std::pair<HeteroGraphPtr, IdArray> SamplingExecutor::ToBlock_(
  const IdArray& target_gnids, const IdArray& src_gnids, const IdArray& dst_gnids) const {
  return ToBlock(
    empty_graph_ref_,
    src_gnids,
    dst_gnids,
    target_gnids,
    null_array_);
}

IdArray SamplingExecutor::GetPartIds(const IdArray& src_gnids) const {
  int64_t num_edges = src_gnids->shape[0];
  
  IdArray src_part_ids = NDArray::Empty({ num_edges }, src_gnids->dtype, src_gnids->ctx);
  const int64_t* src_id_ptr = src_gnids.Ptr<int64_t>();
  int64_t* pid_ptr = src_part_ids.Ptr<int64_t>();
  int num_parts = nid_partitions_.size() - 1;

  parallel_for(0, num_edges, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; i++) {
      int64_t src_id = src_id_ptr[i];
      int part_id = num_parts;
      for (int j = 0; j < num_parts; j++) {
        if (src_id < nid_partitions_[j + 1]) {
          part_id = j;
          break;
        }
      }

      pid_ptr[i] = part_id;
    }
  });

  return src_part_ids;
}

int64_t SamplingExecutor::GetNumLocalTargets(int64_t num_targets) const {
  auto num_assigned_target_per_gpu = GetNumAssignedTargetsPerGpu(
    num_machines_, num_gpus_per_machine_in_group_, num_targets);
  return num_assigned_target_per_gpu[machine_rank_ * num_gpus_per_machine_in_group_ + local_gpu_rank_in_group_];
}

std::tuple<HeteroGraphPtr, IdArray, IdArray> SamplingExecutor::ToDistributedBlock_(
  const IdArray& target_gnids, const IdArray& src_gnids, const IdArray& dst_gnids) const {
  return ToDistributedBlock(
    num_machines_,
    machine_rank_,
    num_gpus_per_machine_in_group_,
    local_gpu_rank_in_group_,
    target_gnids,
    src_gnids,
    GetPartIds(src_gnids),
    dst_gnids);
}

NDArray SamplingExecutor::Pull(const std::string& name, const IdArray& id_arr) const {
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

PackedFunc SamplingExecutor::SendRemotePullReqs(
    const std::string& name,
    const std::vector<int>& part_ids,
    const std::vector<IdArray>& local_nids_list) const {
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

void SamplingExecutor::CopyLocalData(
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

void SamplingExecutor::CopyFetchedData(
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

std::pair<std::vector<IdArray>, std::vector<IdArray>> SamplingExecutor::DistSampling(
  const IdArray& seeds,
  const int fanout) const {

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
      sampled_srcs_list.push_back(ret[2 * i]->data);
      sampled_dsts_list.push_back(ret[2 * i + 1]->data);
    }
  }

  return { sampled_srcs_list, sampled_dsts_list };
}

PackedFunc SamplingExecutor::SendRemoteDistSamplingReqs(
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

std::pair<IdArray, IdArray> SamplingExecutor::LocalSampling(const IdArray& seeds, const int fanout) const {

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
  const int num_edges = src_local_ids->shape[0];

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

  return { src_global_ids, dst_global_ids };
}

}
}
