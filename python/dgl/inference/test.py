# TODO: don't let the test code be here

import time
import dgl
import torch
from .api import *
import random

random.seed(5124)

def create_test_graph(src_gnids, dst_gnids):
    used = set()
    u = []
    v = []
    for i in range(dst_gnids.shape[0]):
        x = src_gnids[i]
        y = dst_gnids[i]
        assert x == y
        u.append(x)
        v.append(y)
        used.add((x, y))

    for i in range(dst_gnids.shape[0] * src_gnids.shape[0] // 4):
        x = src_gnids[random.randint(0, src_gnids.shape[0] - 1)]
        y = dst_gnids[random.randint(0, dst_gnids.shape[0] - 1)]
        if (x, y) not in used:
            used.add((x, y))
            u.append(x)
            v.append(y)

    return dgl.graph((u, v))

def test_split_blocks():
    num_nodes = 4
    num_devices_per_node = 4

    batch_size = 10

    def fn(r):
        return torch.tensor(r, dtype=torch.int64)
    dst_gnids = torch.concat((fn(range(50, 60)), fn(range(11))))
    dst_part_ids = torch.concat((torch.ones(batch_size, dtype=torch.int64) * num_nodes, dst_gnids[batch_size:] % num_nodes))
    src_gnids = torch.concat((fn(range(50, 60)), fn(range(11)), fn(range(21, 39))))
    src_part_ids = torch.concat((torch.ones(batch_size, dtype=torch.int64) * num_nodes, src_gnids[batch_size:] % num_nodes))
    # print(src_gnids)
    # print(src_part_ids)

    sorted_bids, sorted_gnids = sort_dst_ids(num_nodes, num_devices_per_node, batch_size, dst_gnids, dst_part_ids)

    ret0 = extract_src_ids(num_nodes, num_devices_per_node, 0, batch_size, src_gnids, src_part_ids)
    ret1 = extract_src_ids(num_nodes, num_devices_per_node, 1, batch_size, src_gnids, src_part_ids)
    # print([sorted_bids, sorted_gnids])
    # print(ret0)
    # print(ret1)

    # dst_ids = torch.tensor([6, 5, 4, 3, 2, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]) + 10
    # src_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10]) + 10
    # dst_nodes = torch.tensor([0, 1, 2, 3, 4, 5, 6]) + 10
    g = create_test_graph(src_gnids, dst_gnids)
    block = dgl.to_block(g, dst_gnids, src_nodes=src_gnids)
    assert(torch.all(block.srcdata[dgl.NID] == src_gnids))
    assert(torch.all(block.dstdata[dgl.NID] == dst_gnids))

    start = time.time()
    # print(block)
    blocks0 = split_blocks(block, src_part_ids, dst_part_ids, num_nodes, num_devices_per_node, 0, batch_size)
    print(time.time() - start)
    start = time.time()
    blocks1 = split_blocks(block, src_part_ids, dst_part_ids, num_nodes, num_devices_per_node, 1, batch_size)
    print(time.time() - start)

    print(blocks0)
    print(blocks1)

    pass

if __name__ == "__main__":
    test_split_blocks()
