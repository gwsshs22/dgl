import random

import torch

import dgl
from dgl.omega.omega_apis import (
    to_distributed_blocks,
    get_num_assigned_targets_per_gpu)

from test_utils import create_test_data

def test():
    num_existing_nodes = 1000
    num_target_nodes = 4
    num_machines = 1
    num_gpus_per_machine = 4

    num_connecting_edges = 5000

    target_gnids, src_gnids, src_part_ids, dst_gnids = create_test_data(
        num_existing_nodes,
        num_target_nodes,
        num_machines,
        num_connecting_edges,
        random_seed=4132)

    dist_blocks = []    
    for machine_rank in range(num_machines):
        dist_blocks.extend(to_distributed_blocks(
            num_machines,
            machine_rank,
            num_gpus_per_machine,
            target_gnids,
            src_gnids,
            src_part_ids,
            dst_gnids))

    num_targets_per_gpu = get_num_assigned_targets_per_gpu(
        num_machines, num_gpus_per_machine, num_target_nodes)
    u_in_blocks = []
    v_in_blocks = []
    target_ids_in_blocks = []
    other_src_ids_in_blocks = []
    num_srcs = 0
    for gpu_id in range(num_machines * num_gpus_per_machine):
        dist_block = dist_blocks[gpu_id]
        num_srcs += dist_block.num_src_nodes()
        u, v = dist_block.edges('uv')
        u_in_blocks.append(dist_block.srcdata[dgl.NID][u])
        v_in_blocks.append(target_gnids[v])

        num_assigned_targets = num_targets_per_gpu[gpu_id]
        target_ids_in_blocks.append(dist_block.srcdata[dgl.NID][:num_assigned_targets])
        other_src_ids_in_blocks.append(dist_block.srcdata[dgl.NID][num_assigned_targets:])

    u_in_blocks = torch.concat(u_in_blocks)
    v_in_blocks = torch.concat(v_in_blocks)
    target_ids_in_blocks = torch.concat(target_ids_in_blocks)
    other_src_ids_in_blocks = torch.concat(other_src_ids_in_blocks)
    num_total_edges = u_in_blocks.shape[0]

    assert num_srcs <= num_existing_nodes + num_target_nodes, (
        f"num_srcs={num_srcs}, num_existing_nodes + num_target_nodes={num_existing_nodes + num_target_nodes}"
    )
    assert num_total_edges == num_connecting_edges + num_target_nodes
    assert torch.all(u_in_blocks.sort().values == src_gnids.sort().values)
    assert torch.all(v_in_blocks.sort().values == dst_gnids.sort().values)
    assert torch.all(target_ids_in_blocks == target_gnids)
    assert torch.all(other_src_ids_in_blocks < num_existing_nodes)
    assert other_src_ids_in_blocks.shape[0] == other_src_ids_in_blocks.unique().shape[0]
    assert other_src_ids_in_blocks.shape[0] <= num_existing_nodes
    print("Test passed.")

if __name__ == "__main__":
    test()
