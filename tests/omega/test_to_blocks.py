import random

import torch

import dgl
from dgl.omega.omega_apis import (
    to_block,
    to_distributed_block,
    to_distributed_blocks,
    get_num_assigned_targets_per_gpu)

from test_utils import create_test_data

def test():
    device = f"cuda:0"

    num_existing_nodes = 1000
    num_target_nodes = 4
    num_machines = 2
    num_gpus_per_machine = 4

    num_connecting_edges = 5000

    target_gnids, src_gnids, src_part_ids, dst_gnids = create_test_data(
        num_existing_nodes,
        num_target_nodes,
        num_machines,
        num_connecting_edges,
        random_seed=4132)

    block1 = to_block(
        src_gnids,
        dst_gnids,
        target_gnids,
    ).to(device)

    block2 = to_block(
        src_gnids.to(device),
        dst_gnids.to(device),
        target_gnids.to(device)
    )

    assert_blocks_eq(block1, block2)
    print("Test passed.")

def assert_blocks_eq(block1, block2):
    assert torch.all(block1.srcdata[dgl.NID] == block2.srcdata[dgl.NID])
    u1, v1 = block1.edges('uv')
    u2, v2 = block2.edges('uv')
    assert torch.all(u1 == u2)
    assert torch.all(v1 == v2)
    assert block1.num_src_nodes() == block2.num_src_nodes()
    assert block1.num_dst_nodes() == block2.num_dst_nodes()

if __name__ == "__main__":
    test()
