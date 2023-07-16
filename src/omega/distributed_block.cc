#include "distributed_block.h"

#include <dgl/base_heterograph.h>
#include <dgl/immutable_graph.h>

#include <parallel_hashmap/phmap.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace omega {

typedef phmap::flat_hash_map<int64_t, int64_t> IdMap;

std::vector<int64_t> GetNumAssignedTargetsPerGpu(
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

std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> ToDistributedBlocks(
    int num_machines, int machine_rank, int num_gpus_per_machine,
    const IdArray& target_gnids, const IdArray& src_gnids,
    const IdArray& src_part_ids, const IdArray& dst_gnids) {

  std::vector<HeteroGraphPtr> graphs(num_gpus_per_machine);
  std::vector<IdArray> src_gnids_in_blocks(num_gpus_per_machine);
  std::vector<std::vector<int64_t>> src_block_id_lists(num_gpus_per_machine);
  std::vector<std::vector<int64_t>> dst_block_id_lists(num_gpus_per_machine);
  std::vector<IdMap> lhs_node_mappings(num_gpus_per_machine);
  IdMap target_node_gpu_assignment;
  IdMap rhs_node_mapping;

  const int64_t* target_gnids_data = static_cast<int64_t*>(target_gnids->data);
  const int64_t* src_gnids_data = static_cast<int64_t*>(src_gnids->data);
  const int64_t* src_part_ids_data = static_cast<int64_t*>(src_part_ids->data);
  const int64_t* dst_gnids_data = static_cast<int64_t*>(dst_gnids->data);
  const int64_t num_edges = src_gnids->shape[0];
  CHECK_EQ(num_edges, src_part_ids->shape[0]);
  CHECK_EQ(num_edges, dst_gnids->shape[0]);

  const auto num_targets_per_gpu = GetNumAssignedTargetsPerGpu(
    num_machines, num_gpus_per_machine, target_gnids->shape[0]);
  std::vector<int64_t> num_targets_per_gpu_cumsum(1 + num_machines * num_gpus_per_machine);
  num_targets_per_gpu_cumsum[0] = 0;
  for (int i = 0; i < num_targets_per_gpu.size(); i++) {
    num_targets_per_gpu_cumsum[i + 1] = num_targets_per_gpu_cumsum[i] + num_targets_per_gpu[i];
  }

  for (int machine_idx = 0; machine_idx < num_machines; machine_idx++) {
    for (int gpu_idx = 0; gpu_idx < num_gpus_per_machine; gpu_idx++) {
      int global_gpu_idx = machine_idx * num_gpus_per_machine + gpu_idx;
      const auto start_idx = num_targets_per_gpu_cumsum[global_gpu_idx];
      const auto end_idx = num_targets_per_gpu_cumsum[global_gpu_idx + 1];
      for (int i = start_idx; i < end_idx; i++) {
        const auto target_gnid = target_gnids_data[i];
        rhs_node_mapping.insert({target_gnid, rhs_node_mapping.size()});

        if (machine_idx == machine_rank) {
          target_node_gpu_assignment.insert({target_gnid, gpu_idx});
          auto& lhs_node_mapping = lhs_node_mappings[gpu_idx];
          lhs_node_mapping.insert({target_gnid, lhs_node_mapping.size()});
        } else {
          target_node_gpu_assignment.insert({target_gnid, -1});
        }
      }
    }
  }

  CHECK_EQ(rhs_node_mapping.size(), target_gnids->shape[0]);

  for (int64_t i = 0; i < num_edges; i++) {
    int64_t src_part_id = src_part_ids_data[i];
    int64_t src_gnid;
    int assigned_gpu_id;

    if (src_part_id == machine_rank) {
      src_gnid = src_gnids_data[i];
      assigned_gpu_id = src_gnid % num_gpus_per_machine;
    } else if (src_part_id >= num_machines) {
      src_gnid = src_gnids_data[i];
      assigned_gpu_id = target_node_gpu_assignment[src_gnid];
      if (assigned_gpu_id == -1) continue;
    } else {
      continue;
    }

    int64_t dst_gnid = dst_gnids_data[i];
    auto& lhs_node_mapping = lhs_node_mappings[assigned_gpu_id];
    auto insert_ret = lhs_node_mapping.insert({src_gnid, lhs_node_mapping.size()});

    int64_t src_block_id = insert_ret.first->second;
    int64_t dst_block_id = rhs_node_mapping[dst_gnid];

    src_block_id_lists[assigned_gpu_id].push_back(src_block_id);
    dst_block_id_lists[assigned_gpu_id].push_back(dst_block_id);
  }

  for (int gpu_idx = 0; gpu_idx < num_gpus_per_machine; gpu_idx++) {
    IdArray src_ids = NDArray::FromVector(src_block_id_lists[gpu_idx]);
    IdArray dst_ids = NDArray::FromVector(dst_block_id_lists[gpu_idx]);
    auto& lhs_node_mapping = lhs_node_mappings[gpu_idx];

    const auto meta_graph = ImmutableGraph::CreateFromCOO(
        2,
        NDArray::FromVector(std::vector<int64_t>({0})),
        NDArray::FromVector(std::vector<int64_t>({1})));

    std::vector<HeteroGraphPtr> rel_graphs(1);
    std::vector<int64_t> num_nodes_per_type(2);
    num_nodes_per_type[0] = lhs_node_mapping.size();
    num_nodes_per_type[1] = rhs_node_mapping.size();
    rel_graphs[0] = CreateFromCOO(2, num_nodes_per_type[0], num_nodes_per_type[1], src_ids, dst_ids);
    graphs[gpu_idx] = CreateHeteroGraph(meta_graph, rel_graphs, num_nodes_per_type);

    IdArray src_gnids_in_block = NewIdArray(
        lhs_node_mapping.size(), DGLContext{kDGLCPU, 0}, sizeof(int64_t) * 8);
    int64_t* src_gnids_in_block_data = static_cast<int64_t*>(src_gnids_in_block->data);

    for (auto pair: lhs_node_mapping) {
      src_gnids_in_block_data[pair.second] = pair.first;
    }

    src_gnids_in_blocks[gpu_idx] = src_gnids_in_block;
  }

  return std::make_pair(graphs, src_gnids_in_blocks);
}

std::pair<HeteroGraphPtr, IdArray> ToDistributedBlock(
    int num_machines, int machine_rank, int num_gpus_per_machine, int local_gpu_idx,
    const IdArray& target_gnids, const IdArray& src_gnids,
    const IdArray& src_part_ids, const IdArray& dst_gnids) {

  HeteroGraphPtr graph;
  std::vector<int64_t> src_block_ids;
  std::vector<int64_t> dst_block_ids;
  IdMap lhs_node_mapping;
  IdMap target_node_gpu_assignment;
  IdMap rhs_node_mapping;

  const int64_t* target_gnids_data = static_cast<int64_t*>(target_gnids->data);
  const int64_t* src_gnids_data = static_cast<int64_t*>(src_gnids->data);
  const int64_t* src_part_ids_data = static_cast<int64_t*>(src_part_ids->data);
  const int64_t* dst_gnids_data = static_cast<int64_t*>(dst_gnids->data);
  const int64_t num_edges = src_gnids->shape[0];
  CHECK_EQ(num_edges, src_part_ids->shape[0]);
  CHECK_EQ(num_edges, dst_gnids->shape[0]);
  CHECK(local_gpu_idx < num_gpus_per_machine);

  const auto num_targets_per_gpu = GetNumAssignedTargetsPerGpu(
    num_machines, num_gpus_per_machine, target_gnids->shape[0]);
  std::vector<int64_t> num_targets_per_gpu_cumsum(1 + num_machines * num_gpus_per_machine);
  num_targets_per_gpu_cumsum[0] = 0;
  for (int i = 0; i < num_targets_per_gpu.size(); i++) {
    num_targets_per_gpu_cumsum[i + 1] = num_targets_per_gpu_cumsum[i] + num_targets_per_gpu[i];
  }

  for (int machine_idx = 0; machine_idx < num_machines; machine_idx++) {
    for (int gpu_idx = 0; gpu_idx < num_gpus_per_machine; gpu_idx++) {
      int global_gpu_idx = machine_idx * num_gpus_per_machine + gpu_idx;
      const auto start_idx = num_targets_per_gpu_cumsum[global_gpu_idx];
      const auto end_idx = num_targets_per_gpu_cumsum[global_gpu_idx + 1];
      for (int i = start_idx; i < end_idx; i++) {
        const auto target_gnid = target_gnids_data[i];
        rhs_node_mapping.insert({target_gnid, rhs_node_mapping.size()});

        if (machine_idx == machine_rank && local_gpu_idx == gpu_idx) {
          target_node_gpu_assignment.insert({target_gnid, gpu_idx});
          lhs_node_mapping.insert({target_gnid, lhs_node_mapping.size()});
        } else {
          target_node_gpu_assignment.insert({target_gnid, -1});
        }
      }
    }
  }

  CHECK_EQ(rhs_node_mapping.size(), target_gnids->shape[0]);

  for (int64_t i = 0; i < num_edges; i++) {
    int64_t src_part_id = src_part_ids_data[i];
    int64_t src_gnid;
    int assigned_gpu_id;

    if (src_part_id == machine_rank) {
      src_gnid = src_gnids_data[i];
      assigned_gpu_id = src_gnid % num_gpus_per_machine;
      if (assigned_gpu_id != local_gpu_idx) continue;
    } else if (src_part_id >= num_machines) {
      src_gnid = src_gnids_data[i];
      assigned_gpu_id = target_node_gpu_assignment[src_gnid];
      if (assigned_gpu_id == -1) continue;
    } else {
      continue;
    }

    int64_t dst_gnid = dst_gnids_data[i];
    auto insert_ret = lhs_node_mapping.insert({src_gnid, lhs_node_mapping.size()});

    int64_t src_block_id = insert_ret.first->second;
    int64_t dst_block_id = rhs_node_mapping[dst_gnid];

    src_block_ids.push_back(src_block_id);
    dst_block_ids.push_back(dst_block_id);
  }

  IdArray src_ids = NDArray::FromVector(src_block_ids);
  IdArray dst_ids = NDArray::FromVector(dst_block_ids);

  const auto meta_graph = ImmutableGraph::CreateFromCOO(
      2,
      NDArray::FromVector(std::vector<int64_t>({0})),
      NDArray::FromVector(std::vector<int64_t>({1})));

  std::vector<HeteroGraphPtr> rel_graphs(1);
  std::vector<int64_t> num_nodes_per_type(2);
  num_nodes_per_type[0] = lhs_node_mapping.size();
  num_nodes_per_type[1] = rhs_node_mapping.size();
  rel_graphs[0] = CreateFromCOO(2, num_nodes_per_type[0], num_nodes_per_type[1], src_ids, dst_ids);
  graph = CreateHeteroGraph(meta_graph, rel_graphs, num_nodes_per_type);

  IdArray src_gnids_in_block = NewIdArray(
      lhs_node_mapping.size(), DGLContext{kDGLCPU, 0}, sizeof(int64_t) * 8);
  int64_t* src_gnids_in_block_data = static_cast<int64_t*>(src_gnids_in_block->data);

  for (auto pair: lhs_node_mapping) {
    src_gnids_in_block_data[pair.second] = pair.first;
  }

  return std::make_pair(graph, src_gnids_in_block);
}

}
}
