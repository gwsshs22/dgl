#include "graph_api.h"

namespace dgl {
namespace inference {

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

std::vector<IdArray> ExtractSrcIds(int num_nodes,
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

  auto ret_list = std::vector<IdArray>();

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

    ret_list.push_back(sorted_bids_per_gpu);
    ret_list.push_back(sorted_org_ids_per_gpu);
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

  return ret_list;
}

}
}
