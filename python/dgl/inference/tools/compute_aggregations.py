import argparse
import random
import socket
import os
import json

import tqdm
import dgl
import torch
import numpy as np

from dgl.distributed import DistDataLoader
from dgl.inference.models.factory import load_model

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        seeds = torch.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)

        input_nodes = blocks[0].srcdata[dgl.NID]
        batch_inputs = self.g.ndata["features"][input_nodes]
        blocks[0].srcdata["features"] = batch_inputs
        return blocks

class PrecomputedAggregationsHandler():
    def __init__(self):
        self._dist_tensors = {}

    def push(self, g, name, tensor, gnids):
        if name not in self._dist_tensors:
            dist_tensor = dgl.distributed.DistTensor((g.number_of_nodes(),) + tensor.shape[1:], tensor.dtype, name)
            self._dist_tensors[name] = dist_tensor
        else:
            dist_tensor = self._dist_tensors[name]
        dist_tensor[gnids] = tensor.to("cpu")

    def write(self, part_id, part_config, precom_filename):
        config_path = os.path.dirname(part_config)
        precom_filepath = os.path.join(config_path, 'part{}'.format(part_id), precom_filename)
        start_idx, end_idx = self._get_range(part_id, part_config)
        print(f"Write precomuted aggregations to {precom_filepath}, start_idx={start_idx}, end_idx={end_idx}")
        data = {}
        for name, dist_tensor in self._dist_tensors.items():
            data[name] = dist_tensor[start_idx:end_idx]
        dgl.data.utils.save_tensors(precom_filepath, data)
    
    def _get_range(self, part_id, part_config):
        with open(part_config) as conf_f:
            part_metadata = json.load(conf_f)
        return part_metadata["node_map"]["_N"][part_id]

def precompute_aggregations(model, g, local_rank, batch_size, device, part_config, precom_filename):
    model.eval()
    model = model.to(device)

    nodes = dgl.distributed.node_split(np.arange(g.number_of_nodes()),
                                       g.get_partition_book(), force_even=True)
    if g.rank() == 0:
        # Avoid a bug that 0 is not included
        nodes = torch.concat((torch.LongTensor([0]), nodes))
    sampler = NeighborSampler(g, [-1], dgl.distributed.sample_neighbors)
    dataloader = DistDataLoader(
        dataset=nodes,
        batch_size=batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=False,
        drop_last=False)

    first_layer = model.layers[0]

    handler = PrecomputedAggregationsHandler()

    with torch.no_grad():
        for blocks in tqdm.tqdm(dataloader):
            block = blocks[0].to(device)
            src_ids = block.srcdata[dgl.NID]
            dst_ids = block.dstdata[dgl.NID]
            input_features = block.srcdata["features"]
            dst_init_values = first_layer.compute_dst_init_values(block, input_features, dst_ids.shape[0])
            if dst_init_values is not None:
                for name, tensor in dst_init_values.items():
                    handler.push(g, f"div_{name}", tensor, dst_ids)

            aggregations = first_layer.compute_aggregations(block, input_features, dst_init_values)
            for name, tensor in aggregations.items():
                handler.push(g, f"agg_{name}", tensor, dst_ids)
    g.barrier()
    
    if local_rank == 0:
        handler.write(g.get_partition_book().partid, part_config, precom_filename)

def main(args):
    ip_config = args.ip_config
    graph_name = args.graph_name
    part_config = args.part_config
    net_type = args.net_type

    print(socket.gethostname(), 'Initializing DGL dist')
    dgl.distributed.initialize(ip_config, net_type=net_type)

    print(socket.gethostname(), 'Initializing DistGraph')
    g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
    print(socket.gethostname(), 'rank:', g.rank())

    device = torch.device(f"cuda:{args.local_rank}")
    model = load_model(args.model, args.num_inputs, args.num_hiddens, args.num_outputs, args.num_layers, args.heads)
    precompute_aggregations(model, g, args.local_rank, args.batch_size, device, part_config, args.precom_filename)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=412412322)
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--part_config', type=str, required=True)
    parser.add_argument('--ip_config', type=str, required=True)
    parser.add_argument('--net_type', type=str, default="socket", help="'socket' or 'tensorpipe'")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--precom_filename', type=str, required=True)
    # Model parameters
    parser.add_argument("--model", type=str, required=True, choices=["gcn", "sage", "gat"])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_inputs', type=int, default=32)
    parser.add_argument('--num_hiddens', type=int, default=32)
    parser.add_argument('--num_outputs', type=int, default=32)
    parser.add_argument('--heads', type=str, default="8,8", help="The number of attention heads for two-layer gat models")

    args = parser.parse_args()
    print(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    main(args)

# Run this script using original dgl/tools/launch.py
