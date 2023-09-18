import argparse
import random
import socket
import os
import json

import tqdm
import dgl
import torch
import numpy as np

from models import create_model

def precompute_embeddings(model, g, local_rank, num_layers, batch_size, device, part_config, precom_filename):
    model.eval()
    model = model.to(device)

    nodes = dgl.distributed.node_split(
        np.arange(g.num_nodes()),
        g.get_partition_book(),
        force_even=True,
    )

    x = g.ndata["features"]

    precoms = []
    precom_tensor = None
    with torch.no_grad():
        for layer_idx in range(num_layers - 1):
            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h = model.layer_foward(layer_idx, block, h)

                if precom_tensor is None:
                    precom_tensor = dgl.distributed.DistTensor(
                                        (g.num_nodes(), ) + h.shape[1:],
                                        h.dtype,
                                        f"layer_{layer_idx}",
                                        persistent=True,
                                    )

                precom_tensor[output_nodes] = h.cpu()

            g.barrier()

            x = precom_tensor
            precoms.append(precom_tensor)
            precom_tensor = None

    if local_rank == 0:
        part_id = g.get_partition_book().partid
        with open(part_config) as conf_f:
            part_metadata = json.load(conf_f)
        start_idx, end_idx =  part_metadata["node_map"]["_N"][part_id]
        
        config_path = os.path.dirname(part_config)
        precom_filepath = os.path.join(config_path, 'part{}'.format(part_id), precom_filename)

        print(f"Write precomuted aggregations to {precom_filepath}, start_idx={start_idx}, end_idx={end_idx}")
        data = {}
        for layer_idx in range(num_layers - 1):
            data[f"layer_{layer_idx}"] = precoms[layer_idx][start_idx:end_idx]
        dgl.data.utils.save_tensors(precom_filepath, data)

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

    if args.model == 'gat':
        gat_heads = [int(h) for h in args.gat_heads.split(",")]
        assert len(gat_heads) == args.num_layers
    else:
        gat_heads = None

    model = create_model(args.model, args.num_inputs, args.num_hiddens, args.num_outputs, args.num_layers, gat_heads)
    precompute_embeddings(model, g, args.local_rank, args.num_layers, args.batch_size, device, part_config, args.precom_filename)
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
    parser.add_argument("--gat_heads", default="8,8")

    args = parser.parse_args()
    print(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    main(args)

# Run this script using original dgl/tools/launch.py
