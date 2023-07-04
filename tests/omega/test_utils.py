import random

import torch

def create_test_data(
    num_existing_nodes,
    num_target_nodes,
    num_machines,
    num_connecting_edges,
    random_seed):

    random.seed(random_seed)
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

    return target_gnids, src_gnids, src_part_ids, dst_gnids
