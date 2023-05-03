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

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import socket

from models import create_model
from utils import Timer

def inference(args, dataloader, g, model, device, fanouts):
    def sample_blocks(g, seeds, fanouts, device):
        blocks = []
        for fanout in fanouts:
            with timer.measure("sample_neighbors"):
                frontier = dgl.distributed.sample_neighbors(g, seeds.cpu(), fanout, replace=False)
            
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
                    blocks = sample_blocks(g, seeds, fanouts, device)

                with timer.measure("fetch_feats"):
                    input_nodes = blocks[0].srcdata[dgl.NID].cpu()
                    batch_inputs = g.ndata["features"][input_nodes]

                with timer.measure("copy_feats", cuda_sync=True, device=device):
                    batch_inputs = batch_inputs.to(device)
                
                with timer.measure("execute", cuda_sync=True, device=device):
                    batch_pred = model(blocks, batch_inputs)

                for i in range(4):
                    timer.add(f"n_nodes_{i}", blocks[i].num_src_nodes() if i < args.num_layers else 0)
                    timer.add(f"n_edges_{i}", blocks[i].num_edges() if i < args.num_layers else 0)

                values_arr.append(timer.get_values())

                if (step + 1) == NUM_STEPS:
                    break
 
    return values_arr[NUM_STEPS:]

def inference_with_precoms(args, dataloader, g, model, device, fanouts):
    def sample_blocks(g, seeds, fanouts, device):
        blocks = []
        for fanout in fanouts:
            with timer.measure("sample_neighbors"):
                frontier = dgl.distributed.sample_neighbors(g, seeds.cpu(), fanout, replace=False)
            
            with timer.measure("create_mfgs", cuda_sync=True, device=device):
                block = dgl.to_block(frontier, seeds)
                blocks.insert(0, block)
        
        with timer.measure("copy_mfgs", cuda_sync=True, device=device):
            blocks = [b.to(device) for b in blocks]

        return blocks

    timer = Timer()
    values_arr = []

    precoms_dist_tensors = [g.ndata["features"]]
    for layer_idx in range(args.num_layers - 1):
        precoms_dist_tensors.append(g.ndata[f"layer_{layer_idx}"])

    NUM_RUNS = args.num_runs + 1 # First one is warm-up
    NUM_STEPS = args.num_steps
    for run in range(NUM_RUNS):
        with th.no_grad():
            # for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            for step, seeds in enumerate(dataloader):
                timer.clear()
                inputs = []
                with timer.measure("mfgs"):
                    blocks = sample_blocks(g, seeds, fanouts, device)

                with timer.measure("fetch_feats"):
                    for layer_idx in range(args.num_layers):
                        if layer_idx == 0:
                            input_nodes = blocks[layer_idx].srcdata[dgl.NID]
                        else:
                            block = blocks[layer_idx]
                            input_nodes = block.srcdata[dgl.NID][block.num_dst_nodes():]
                        inputs.append(precoms_dist_tensors[layer_idx][input_nodes])

                with timer.measure("copy_feats", cuda_sync=True, device=device):
                    inputs = [i.to(device) for i in inputs]
                
                with timer.measure("execute", cuda_sync=True, device=device):
                    h = inputs[0]
                    for layer_idx in range(args.num_layers):
                        h = model.layer_foward(layer_idx, blocks[layer_idx], h)
                        if layer_idx != args.num_layers - 1:
                            h = th.concat((h, inputs[layer_idx + 1]))

                for i in range(4):
                    timer.add(f"n_nodes_{i}", blocks[i].num_src_nodes() if i < args.num_layers else 0)
                    timer.add(f"n_edges_{i}", blocks[i].num_edges() if i < args.num_layers else 0)

                values_arr.append(timer.get_values())

                if (step + 1) == NUM_STEPS:
                    break
 
    return values_arr[NUM_STEPS:]

def main(args):
    print(socket.gethostname(), 'Initializing DGL dist')
    use_precoms = args.use_precoms > 0
    dgl.distributed.initialize(args.ip_config, net_type=args.net_type, load_precoms=use_precoms, precom_path=args.precom_path)
    if not args.standalone:
        print(socket.gethostname(), 'Initializing DGL process group')
        th.distributed.init_process_group(backend=args.backend)
    print(socket.gethostname(), 'Initializing DistGraph')
    
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    rank = g.rank()
    print(socket.gethostname(), 'rank:', rank)

    if g.rank() != args.target_rank:
        print(f"rank={rank} call barrier")
        g.barrier()
        print(f"rank={rank} end barrier")
        return

    device = f"cuda:{args.local_rank}"

    in_size = args.n_inputs
    out_size = args.n_classes

    gat_heads = [int(h) for h in args.gat_heads.split(",")]
    model = create_model(args.gnn, in_size, args.num_hiddens, out_size, args.num_layers, gat_heads)
    print(f"rank={rank} Load model")
    model = model.to(device)
    model.eval()

    shuffle = True
    # Create sampler
    print(f"rank={rank} Create sampler")
    fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    fanouts = [-1 if f == 0 else f for f in fanouts]

    dataloader = DataLoader(
        th.arange(g.number_of_nodes()),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    print(f"rank={rank} Start iterating")

    if not use_precoms:
        values_arr = inference(args, dataloader, g, model, device, fanouts)
    else:
        values_arr = inference_with_precoms(args, dataloader, g, model, device, fanouts)

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
    kvs.append(("gnn", args.gnn))
    kvs.append(("bs", args.batch_size))
    kvs.append(("precoms", int(use_precoms)))
    kvs.append(("fo", fanout_str))

    mfgs = get_mean("mfgs")
    fetch_feats = get_mean("fetch_feats")
    copy_feats = get_mean("copy_feats")
    execute = get_mean("execute")
    total = mfgs + fetch_feats + copy_feats + execute
    kvs.append(("total", total))
    kvs.append(("mfgs", mfgs))
    kvs.append(("fetch_feats", fetch_feats))
    kvs.append(("copy_feats", copy_feats))
    kvs.append(("execute", execute))

    kvs.append(("sample_neighbors", get_mean("sample_neighbors")))
    kvs.append(("create_mfgs", get_mean("create_mfgs")))
    kvs.append(("copy_mfgs", get_mean("copy_mfgs")))

    for i in range(4):
        kvs.append((f"n_nodes_{i}", get_mean(f"n_nodes_{i}")))

    for i in range(4):
        kvs.append((f"n_edges_{i}", get_mean(f"n_edges_{i}")))

    kvs.append(("gpu_mem_alloc", gpu_mem_alloc))

    output_path = str(Path(args.output_dir) / f"{args.graph_name}_{args.gnn}_precom{use_precoms}_{fanout_str}_bs{args.batch_size}.txt")
    with open(output_path, "w") as f:
        keys = [kv[0] for kv in kvs]
        values = [str(kv[1]) for kv in kvs]
        f.write(",".join(keys) + "\n")
        f.write(",".join(values) + "\n")

    g.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_clients', type=int, help='The number of clients')
    parser.add_argument('--backend', type=str, default='gloo', help='pytorch distributed backend')
    parser.add_argument('--num_gpus', type=int, default=-1,
                        help="the number of GPU device. Use -1 for CPU training")

    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--fan_out', type=str, default='10,25')

    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    parser.add_argument('--pad-data', default=False, action='store_true',
                        help='Pad train nid to the same length across machine, to ensure num of batches to be the same.')
    parser.add_argument('--net_type', type=str, default='socket',
                        help="backend net type, 'socket' or 'tensorpipe'")
    
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=512341223)
    parser.add_argument("--target_rank", type=int)
    parser.add_argument("--batch_size", type=int, default=1000)

    parser.add_argument("--use_precoms", type=int, default=0)
    parser.add_argument("--precom_path", type=str, default="")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_hiddens", type=int, default=16)
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument('--n_inputs', type=int, help='the number of inputs')
    parser.add_argument("--gnn", choices=["gcn", "sage", "gat"])
    # Sage configuration
    parser.add_argument("--sage_aggr", choices=["mean", "pool"], default="mean")
    # Gat configuration
    parser.add_argument("--gat_heads", default="8,8")
    parser.add_argument("--gnn_last_mean", action='store_true')
    args = parser.parse_args()

    th.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    print(args)
    main(args)

