#include "graph_api.h"

#include <dgl/runtime/parallel_for.h>
#include <parallel_hashmap/phmap.h>
#include <dgl/immutable_graph.h>

namespace dgl {
namespace inference {

std::pair<std::vector<IdArray>, std::vector<IdArray>> SplitLocalEdges(int num_nodes,
                                                                     const IdArray& global_src,
                                                                     const IdArray& global_dst,
                                                                     const IdArray& global_src_part_ids) {
  int64_t* part_ptr = (int64_t*) global_src_part_ids->data;
  auto init_counter = std::vector<int64_t>(num_nodes);
  auto num_edges_per_part = runtime::parallel_reduce(0, global_src_part_ids->shape[0], 1, init_counter, 
    [=](size_t b, size_t e, std::vector<int64_t> counter) {
      for (int i = b; i < e; i++) {
        counter[*(part_ptr + i)]++;
      }

      return counter;
    },
    [](std::vector<int64_t> a, std::vector<int64_t> b) {
      for (int i = 0; i < a.size(); i++) {
        a[i] += b[i];
      }

      return a;
    });

  auto global_src_list = std::vector<IdArray>();
  auto global_dst_list = std::vector<IdArray>();

  auto global_src_ptr_list = std::vector<int64_t*>();
  auto global_dst_ptr_list = std::vector<int64_t*>();

  for (int i = 0; i < num_nodes; i++) {
    global_src_list.push_back(aten::NewIdArray(num_edges_per_part[i]));
    global_src_ptr_list.push_back((int64_t*)global_src_list[i]->data);
    global_dst_list.push_back(aten::NewIdArray(num_edges_per_part[i]));
    global_dst_ptr_list.push_back((int64_t*)global_dst_list[i]->data);
  }

  int64_t* src_ptr = (int64_t*) global_src->data;
  int64_t* dst_ptr = (int64_t*) global_dst->data;
  part_ptr = (int64_t*) global_src_part_ids->data;

  for (int i = 0; i < global_src->shape[0]; i++) {
    auto part = *part_ptr++;
    *global_src_ptr_list[part]++ = *src_ptr++;
    *global_dst_ptr_list[part]++ = *dst_ptr++;
  }

  return std::make_pair(global_src_list, global_dst_list);
}

std::tuple<IdArray, IdArray, IdArray> SortDstIds(int num_nodes,
                                                 int num_devices_per_node,
                                                 int batch_size,
                                                 const IdArray& org_ids,
                                                 const IdArray& part_ids,
                                                 const IdArray& part_id_counts) {
  CHECK(org_ids.NumElements() == part_ids.NumElements());
  const size_t num_dst_nodes = org_ids->shape[0];
  auto sorted_bids = aten::NewIdArray(num_dst_nodes); // sorted ids in block
  auto sorted_org_ids = aten::NewIdArray(num_dst_nodes);

  std::vector<int64_t> num_part_ids;
  for (int i = 0; i < num_nodes; i++) {
    num_part_ids.push_back(((int64_t*)(part_id_counts->data))[i]);
  }

  std::vector<int64_t> num_assigned_batch_inputs_per_gpus;
  std::vector<int64_t> num_assigned_inputs_per_gpus;
  for (int i = 0; i < num_nodes; i++) {
    int64_t num_assigned_to_machine = batch_size / num_nodes;
    if (i < batch_size % num_nodes) {
      num_assigned_to_machine++;
    }

    for (int j = 0; j < num_devices_per_node; j++) {
      int64_t num_assigned_to_gpu = num_assigned_to_machine / num_devices_per_node;
      if (j < num_assigned_to_machine % num_devices_per_node) {
        num_assigned_to_gpu++;
      }
      num_assigned_batch_inputs_per_gpus.push_back(num_assigned_to_gpu);
    }
  }

  int64_t** sorted_bid_ptrs = (int64_t **)malloc(sizeof (int64_t*) * num_nodes * num_devices_per_node);
  int64_t** sorted_org_ids_ptrs = (int64_t **)malloc(sizeof (int64_t*) * num_nodes * num_devices_per_node);

  size_t gpu_idx = 0;
  size_t cumsum = 0;
  for (int i = 0; i < num_nodes; i++) {
    const size_t num_assigned = num_part_ids[i];
    for (int j = 0; j < num_devices_per_node; j++) {
      size_t num_assigned_to_gpu = num_assigned / num_devices_per_node;
      if (j < num_assigned % num_devices_per_node) {
        num_assigned_to_gpu++;
      }

      num_assigned_to_gpu += num_assigned_batch_inputs_per_gpus[i * num_devices_per_node + j];

      sorted_bid_ptrs[gpu_idx] = ((int64_t*)sorted_bids->data) + cumsum;
      sorted_org_ids_ptrs[gpu_idx] = ((int64_t*)sorted_org_ids->data) + cumsum;

      num_assigned_inputs_per_gpus.push_back(num_assigned_to_gpu);
      cumsum += num_assigned_to_gpu;
      gpu_idx++;
    }
  }

  CHECK(gpu_idx == num_nodes * num_devices_per_node);
  CHECK(cumsum == num_dst_nodes);

  int idx = 0;
  int64_t *org_id_ptr = (int64_t*) org_ids->data;
  int64_t *part_id_ptr = (int64_t*) part_ids->data;
  for (int i = 0; i < num_nodes * num_devices_per_node; i++) {
    for (int j = 0; j < num_assigned_batch_inputs_per_gpus[i]; j++) {
      *sorted_bid_ptrs[i]++ = idx++;
      *sorted_org_ids_ptrs[i]++ = *org_id_ptr++;
      part_id_ptr++;
    }
  }

  auto target_gpu_per_machine = new int[num_nodes];

  for (int i = 0; i < num_nodes; i++) {
    target_gpu_per_machine[i] = i * num_devices_per_node;
  }

  while (idx < num_dst_nodes) {
    auto part_id = *part_id_ptr++;
    auto gpu_idx = target_gpu_per_machine[part_id];
    if ((gpu_idx + 1) % num_devices_per_node == 0) {
      target_gpu_per_machine[part_id] = gpu_idx + 1 - num_devices_per_node;
    } else {
      target_gpu_per_machine[part_id] = gpu_idx + 1;
    }

    *sorted_bid_ptrs[gpu_idx]++ = idx++;
    *sorted_org_ids_ptrs[gpu_idx]++ = *org_id_ptr++;
  }

  free(sorted_bid_ptrs);
  free(sorted_org_ids_ptrs);
  delete target_gpu_per_machine;
  return std::make_tuple(std::move(sorted_bids), std::move(sorted_org_ids), NDArray::FromVector(num_assigned_inputs_per_gpus));
}

std::pair<std::vector<IdArray>, std::vector<IdArray>> ExtractSrcIds(int num_nodes,
                                                                    int num_devices_per_node,
                                                                    int node_rank,
                                                                    int batch_size,
                                                                    const IdArray& org_ids,
                                                                    const IdArray& part_ids,
                                                                    const IdArray& part_id_counts) {
  
  CHECK(org_ids.NumElements() == part_ids.NumElements());
  const size_t num_src_nodes = org_ids->shape[0];

  std::vector<int64_t> num_part_ids;
  for (int i = 0; i < num_nodes; i++) {
    num_part_ids.push_back(((int64_t*)(part_id_counts->data))[i]);
  }

  std::vector<int64_t> num_assigned_batch_inputs_per_machines;
  std::vector<int64_t> num_assigned_batch_inputs_per_gpus;
  for (int i = 0; i < num_nodes; i++) {
    int64_t num_assigned_to_machine = batch_size / num_nodes;
    if (i < batch_size % num_nodes) {
      num_assigned_to_machine++;
    }

    num_assigned_batch_inputs_per_machines.push_back(num_assigned_to_machine);
    for (int j = 0; j < num_devices_per_node; j++) {
      int64_t num_assigned_to_gpu = num_assigned_to_machine / num_devices_per_node;
      if (j < num_assigned_to_machine % num_devices_per_node) {
        num_assigned_to_gpu++;
      }
      num_assigned_batch_inputs_per_gpus.push_back(num_assigned_to_gpu);
    }
  }

  auto sorted_bids_list = std::vector<IdArray>();
  auto sorted_orig_ids_list = std::vector<IdArray>();

  int64_t** sorted_bid_ptrs = (int64_t **)malloc(sizeof (int64_t*) * num_devices_per_node);
  int64_t** sorted_org_ids_ptrs = (int64_t **)malloc(sizeof (int64_t*) * num_devices_per_node);

  const size_t num_assigned = num_part_ids[node_rank];
  for (int gpu_idx = 0; gpu_idx < num_devices_per_node; gpu_idx++) {
    size_t num_assigned_to_gpu = num_assigned / num_devices_per_node;
    if (gpu_idx < num_assigned % num_devices_per_node) {
      num_assigned_to_gpu++;
    }

    num_assigned_to_gpu += num_assigned_batch_inputs_per_gpus[node_rank * num_devices_per_node + gpu_idx];

    auto sorted_bids_per_gpu = aten::NewIdArray(num_assigned_to_gpu);
    auto sorted_org_ids_per_gpu = aten::NewIdArray(num_assigned_to_gpu);

    sorted_bid_ptrs[gpu_idx] = ((int64_t*)sorted_bids_per_gpu->data);
    sorted_org_ids_ptrs[gpu_idx] = ((int64_t*)sorted_org_ids_per_gpu->data);

    sorted_bids_list.push_back(sorted_bids_per_gpu);
    sorted_orig_ids_list.push_back(sorted_org_ids_per_gpu);
  }

  int idx = 0;
  int64_t *org_id_ptr = (int64_t*) org_ids->data;
  int64_t *part_id_ptr = (int64_t*) part_ids->data;

  for (int machine_idx = 0; machine_idx < num_nodes; machine_idx++) {
    for (int gpu_idx = 0; gpu_idx < num_devices_per_node; gpu_idx++) {
      int global_gpu_idx = machine_idx * num_devices_per_node + gpu_idx;
      if (machine_idx == node_rank) {
        for (int i = 0; i < num_assigned_batch_inputs_per_gpus[global_gpu_idx]; i++) {
          *sorted_bid_ptrs[gpu_idx]++ = idx++;
          *sorted_org_ids_ptrs[gpu_idx]++ = *org_id_ptr++;
          part_id_ptr++;
        }
      } else {
        idx += num_assigned_batch_inputs_per_gpus[global_gpu_idx];
        org_id_ptr += num_assigned_batch_inputs_per_gpus[global_gpu_idx];
        part_id_ptr += num_assigned_batch_inputs_per_gpus[global_gpu_idx];
      }
    }
  }

  CHECK(idx == batch_size);

  int target_gpu = 0;
  while (idx < num_src_nodes) {
    auto part_id = *part_id_ptr++;

    if (part_id == node_rank) {
      *sorted_bid_ptrs[target_gpu]++ = idx++;
      *sorted_org_ids_ptrs[target_gpu]++ = *org_id_ptr++;
      target_gpu++;
      target_gpu %= num_devices_per_node;
    } else {
      idx++;
      org_id_ptr++;
    }
  }

  free(sorted_bid_ptrs);
  free(sorted_org_ids_ptrs);

  return std::make_pair(std::move(sorted_bids_list), std::move(sorted_orig_ids_list));
}

std::vector<HeteroGraphPtr> SplitBlocks(const HeteroGraphRef& graph_ref,
                                        int num_devices_per_node,
                                        const std::vector<IdArray>& sorted_src_bids_list,
                                        const IdArray& sorted_dst_bids) {
  const auto meta_graph = graph_ref->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const int64_t num_ntypes = 1;

  const IdArray new_dst = aten::Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = dgl::ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, new_dst);
  auto dst_mapping = phmap::flat_hash_map<int64_t, int64_t>();
  auto num_dst = sorted_dst_bids->shape[0];
  dst_mapping.reserve(num_dst);
  auto dst_bid_ptr = (int64_t*) sorted_dst_bids->data;
  for (int i = 0; i < num_dst; i++) {
    dst_mapping.insert({ *dst_bid_ptr++, i});
  }

  auto splitted_graphs = std::vector<HeteroGraphPtr>(num_devices_per_node);
  size_t b = 0;
  size_t e = num_devices_per_node;
  for (size_t gpu_idx = b; gpu_idx < e; gpu_idx++) {

    std::vector<int64_t> num_nodes_per_type(2);
    std::vector<HeteroGraphPtr> rel_graphs;
    auto src_bids = sorted_src_bids_list[gpu_idx];
    auto num_srcs = src_bids->shape[0];

    auto src_mapping = phmap::flat_hash_map<int64_t, int64_t>();
    src_mapping.reserve(num_srcs);
    auto src_bid_ptr = (int64_t*) src_bids->data;
    for (int i = 0; i < num_srcs; i++) {
      src_mapping.insert({ *src_bid_ptr++, i });
    }

    auto edges = graph_ref->OutEdges(0, sorted_src_bids_list[gpu_idx]);

    auto e_src = edges.src;
    auto e_src_ptr = (int64_t*) e_src->data;
    auto e_dst = edges.dst;
    auto e_dst_ptr = (int64_t*) e_dst->data;
    auto new_src = aten::NewIdArray(e_src->shape[0]);
    auto new_src_ptr = (int64_t*) new_src->data;
    auto new_dst = aten::NewIdArray(e_dst->shape[0]);
    auto new_dst_ptr = (int64_t*) new_dst->data;

    auto num_edges = e_src->shape[0];

    runtime::parallel_for(0, num_edges, [=, &src_mapping, &dst_mapping](size_t b1, size_t e1) {
      for (int i = b1; i < e1; i++) {
        *(new_src_ptr + i) = src_mapping[*(e_src_ptr + i)];
        *(new_dst_ptr + i) = dst_mapping[*(e_dst_ptr + i)];
      }
    });

    num_nodes_per_type[0] = src_mapping.size();
    num_nodes_per_type[1] = dst_mapping.size();

    rel_graphs.push_back(CreateFromCOO(
        2, src_mapping.size(), dst_mapping.size(),
        new_src, new_dst));
    splitted_graphs[gpu_idx] = CreateHeteroGraph(
      new_meta_graph, rel_graphs, num_nodes_per_type);
  }

  return splitted_graphs;
}

}
}
