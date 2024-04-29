import argparse
import os
import json
import sys
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch

import dgl
from dgl import function as fn

from dgl.omega.omega_apis import to_block, sample_edges
from omega.utils import get_dataset_config
from omega.utils import load_traces
from omega.utils import cal_metrics
from omega.models import load_model_from

gcn_both_norm = False
def main(args):
    global gcn_both_norm

    device = f"cuda:{args.local_rank}"
    model, training_config, dataset_config = load_model_from(args.training_dir)
    training_id = training_config["id"]
    part_config_path = Path(args.part_config)
    part_config = json.loads(part_config_path.read_text())
    part_config_dir = part_config_path.parent

    gcn_both_norm = (training_config["gnn"] == "gcn" or training_config["gnn"] == "gcn2") and training_config["gcn_norm"] == "both"
    print(f"gcn_both_norm={gcn_both_norm}", file=sys.stderr)
    traces = load_traces(args.trace_dir)

    assert training_config["graph_name"] == part_config["graph_name"]
    assert part_config["num_parts"] == 1
    model = model.to(device)
    model.eval()
    num_layers = training_config["num_layers"]
    num_hiddens = training_config["num_hiddens"]

    g = dgl.load_graphs(str(part_config_dir / "part0" / "graph.dgl"))[0][0]

    node_feats = dgl.data.load_tensors(str(part_config_dir / "part0" / "node_feat.dgl"))
    features = node_feats["_N/features"]
    labels = node_feats["_N/labels"]

    compute_times = []
    for trace in traces:
        compute_time = compute_full_blocks(g, features, device, model, num_layers, trace)
        compute_times.append(compute_time)
    
    print(f"Average compute time = {np.mean(compute_times)}")

def compute_full_blocks(g, features, device, model, num_layers, trace):
    batch_size = trace.target_gnids.shape[0]
    last_block = to_block(trace.src_gnids, trace.dst_gnids, trace.target_gnids)
    blocks = [last_block]

    for _ in range(num_layers - 1):
        seeds = blocks[0].srcdata[dgl.NID][batch_size:]
        frontier = dgl.sampling.sample_neighbors(g, seeds, -1)
        u, v = frontier.edges()
        non_self_edge_mask = trace.dst_gnids != trace.src_gnids
        edge_mask = trace.src_gnids < trace.target_gnids.min()

        block = to_block(
            torch.concat((trace.src_gnids, trace.dst_gnids[edge_mask], u)),
            torch.concat((trace.dst_gnids, trace.src_gnids[edge_mask], v)),
            blocks[0].srcdata[dgl.NID]
        )

        blocks.insert(0, block)

    if gcn_both_norm:
        for block in blocks:
            block.set_out_degrees(torch.concat((
                blocks[-2].out_degrees()[:batch_size],
                g.out_degrees(block.srcdata[dgl.NID][batch_size:]))
            ))

    input_h = features[blocks[0].srcdata[dgl.NID][batch_size:]]
    input_h = torch.concat((trace.target_features, input_h)).to(device)
    blocks = [b.to(device) for b in blocks]

    torch.cuda.synchronize()

    num_repeat = 11
    with torch.no_grad():
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_repeat)]

        for r in range(num_repeat):
            start_events[r].record()
            h = model(blocks, input_h)
            h = h.cpu()
            end_events[r].record()
    
    torch.cuda.synchronize()
    times = [start_events[r].elapsed_time(end_events[r]) for r in range(num_repeat)]
    return np.mean(times[1:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--part_config', type=str, required=True)
    parser.add_argument('--training_dir', type=str, required=True)
    parser.add_argument('--trace_dir', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=451241)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    print(args)
    main(args)
