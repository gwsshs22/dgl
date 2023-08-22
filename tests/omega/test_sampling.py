import os

os.environ["OMP_NUM_THREADS"] = "2"

import torch

import dgl
from dgl.omega.omega_apis import sample_edges

def test():
    target_gnids = torch.tensor([17, 18, 19, 20], dtype=torch.int64)
    src_gnids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 4, 3, 2], dtype=torch.int64)
    dst_gnids = torch.tensor([20, 18, 18, 19, 17, 17, 20, 20, 17, 19, 18, 17, 18, 20, 19, 20, 17, 20, 20, 19], dtype=torch.int64)

    assert src_gnids.shape[0] == dst_gnids.shape[0]
    ret = sample_edges(target_gnids, src_gnids, dst_gnids, [2, 4])
    print(ret)


if __name__ == "__main__":
    test()
