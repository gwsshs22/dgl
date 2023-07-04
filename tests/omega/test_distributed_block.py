import random

import torch

import dgl
from dgl.omega.omega_apis import (
    to_distributed_blocks,
    get_num_assigned_targets_per_gpu)

def test():
    num_existing_nodes = 205
    num_target_nodes = 20
    num_machines = 4
    num_gpus_per_machine = 2

    num_connecting_edges = 1024

    existing_gnids = torch.arange(num_existing_nodes, dtype=torch.int64)
    target_gnids = torch.arange(
        num_existing_nodes, num_existing_nodes + num_target_nodes, dtype=torch.int64)

    src_gnids = []
    src_part_ids = []
    dst_gnids = []

    for _ in range(num_connecting_edges):
        src_gnid = random.randint(0, num_existing_nodes - 1)
        dst_gnid = random.randint(num_existing_nodes, num_existing_nodes + num_target_nodes - 1)
        src_gnids.append(src_gnid)
        src_part_id = src_gnid // (num_existing_nodes // num_machines)
        if src_part_id == num_machines:
            src_part_id -= 1 
        src_part_ids.append(src_part_id)
        
        dst_gnids.append(dst_gnid)

    for target_node_gnid in range(num_existing_nodes, num_existing_nodes + num_target_nodes):
        src_gnids.append(target_node_gnid)
        src_part_ids.append(num_machines + 1)
        dst_gnids.append(target_node_gnid)

    src_gnids = torch.tensor(src_gnids, dtype=torch.int64)
    src_part_ids = torch.tensor(src_part_ids, dtype=torch.int64)
    dst_gnids = torch.tensor(dst_gnids, dtype=torch.int64)

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
    for gpu_id in range(num_machines * num_gpus_per_machine):
        dist_block = dist_blocks[gpu_id]
        u, v = dist_block.edges('uv')
        u_in_blocks.append(dist_block.srcdata[dgl.NID][u])
        v_in_blocks.append(dist_block.dstdata[dgl.NID][v])

        num_assigned_targets = num_targets_per_gpu[gpu_id]
        target_ids_in_blocks.append(dist_block.srcdata[dgl.NID][:num_assigned_targets])
        other_src_ids_in_blocks.append(dist_block.srcdata[dgl.NID][num_assigned_targets:])

    u_in_blocks = torch.concat(u_in_blocks)
    v_in_blocks = torch.concat(v_in_blocks)
    target_ids_in_blocks = torch.concat(target_ids_in_blocks)
    other_src_ids_in_blocks = torch.concat(other_src_ids_in_blocks)
    num_total_edges = u_in_blocks.shape[0]
    assert num_total_edges == num_connecting_edges + num_target_nodes

    assert torch.all(u_in_blocks.sort().values == src_gnids.sort().values)
    assert torch.all(v_in_blocks.sort().values == dst_gnids.sort().values)
    assert torch.all(target_ids_in_blocks == target_gnids)
    assert torch.all(other_src_ids_in_blocks < num_existing_nodes)
    assert other_src_ids_in_blocks.shape[0] == other_src_ids_in_blocks.unique().shape[0]
    assert other_src_ids_in_blocks.shape[0] < num_existing_nodes

if __name__ == "__main__":
    test()