import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
from pathlib import Path
import numpy as np
from functools import wraps
import tqdm

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.distributed import DistDataLoader
from dgl.sampling import sample_neighbors as local_sample_neighbors
from dgl.omega.dist_context import set_nccl_group
from dgl.omega.dist_sample import dist_sample_neighbors
from dgl.omega.omega_apis import to_distributed_block
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import socket

from utils import Timer, load_graph

def sampling(args, dataloader, g, device, fanouts):

    def sample_blocks(g, seeds, fanouts, device):
        blocks = []
        for fanout in fanouts:
            with timer.measure("sample_neighbors"):
                frontier = local_sample_neighbors(g, seeds, fanout, replace=False)

            with timer.measure("create_mfgs", cuda_sync=True, device=device):
                block = dgl.to_block(frontier, seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)

        with timer.measure("copy_mfgs", cuda_sync=True, device=device):
            blocks = [b.to(device) for b in blocks]
        return blocks

    timer = Timer()
    values_arr = []

    NUM_RUNS = args.num_runs + 1 # First one is warm-up
    NUM_STEPS = args.num_steps
    for run in range(NUM_RUNS):
        with th.no_grad():
            # for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            for step, seeds in enumerate(dataloader):
                timer.clear()
                with timer.measure("mfgs"):
                    if args.sampling_device == "gpu":
                        seeds = seeds.to(device)
                    blocks = sample_blocks(g, seeds, fanouts, device)

                values_arr.append(timer.get_values())

                if (step + 1) == NUM_STEPS:
                    break
 
    return values_arr[NUM_STEPS:]

def main(args):
    g = load_graph(args.graph_name)
    device = f"cuda:{args.local_rank}"
    if args.sampling_device == "gpu":
        g = g.to(device)

    shuffle = True
    # Create sampler

    fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    fanouts = [-1 if f == 0 else f for f in fanouts]

    dataloader = DataLoader(
        th.arange(g.number_of_nodes()),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    values_arr = sampling(args, dataloader, g, device, fanouts)

    def get_mean(name):
        arr = []
        for t in values_arr:
            arr.append(t[name])
        return np.mean(arr)

    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
    fanout_str =  str(fanouts).replace(' ', '').replace("'", '').replace("[", "").replace("]", "").replace(",", "_").replace("-1", "f")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    kvs = []
    kvs.append(("graph", args.graph_name))
    kvs.append(("bs", args.batch_size))
    kvs.append(("sampling_device", args.sampling_device))
    kvs.append(("fo", fanout_str))

    mfgs = get_mean("mfgs")
    kvs.append(("mfgs", mfgs))
    kvs.append(("sample_neighbors", get_mean("sample_neighbors")))
    kvs.append(("create_mfgs", get_mean("create_mfgs")))
    kvs.append(("copy_mfgs", get_mean("copy_mfgs")))
    kvs.append(("gpu_mem_alloc", gpu_mem_alloc))

    output_path = str(Path(args.output_dir) / f"{args.graph_name}_sd{args.sampling_device}_{fanout_str}_bs{args.batch_size}.txt")
    with open(output_path, "w") as f:
        keys = [kv[0] for kv in kvs]
        values = [str(kv[1]) for kv in kvs]
        f.write(",".join(keys) + "\n")
        f.write(",".join(values) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--fan_out', type=str, default='10,25')
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=512341223)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--sampling_device", type=str, default="cpu", choices=["cpu", "gpu"])

    args = parser.parse_args()

    th.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    print(args)
    main(args)
