import argparse
import os
import json
import sys
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt

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

    if dataset_config.multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    def get_acc_fn(y_true, y_pred):
        f1mic, _ = cal_metrics(y_true, y_pred, dataset_config.multilabel)
        return f1mic

    assert training_config["graph_name"] == part_config["graph_name"]
    assert part_config["num_parts"] == 1
    model = model.to(device)
    model.eval()
    num_layers = training_config["num_layers"]
    num_hiddens = training_config["num_hiddens"]

    g = dgl.load_graphs(str(part_config_dir / "part0" / "graph.dgl"))[0][0]
    pe_data_path = part_config_dir / "part0" / "data" / training_id / "tensors.pth"

    if pe_data_path.exists():
        pe_data = torch.load(pe_data_path)
        pe_provider = PrecomputedPEProvider([p.to(device) for p in pe_data["pes"]])
    else:
        pe_provider = None
    assert pe_provider is not None
    node_feats = dgl.data.load_tensors(str(part_config_dir / "part0" / "node_feat.dgl"))
    features = node_feats["_N/features"]
    num_nodes = features.shape[0]
    num_eval_traces = args.num_eval_traces if args.num_eval_traces > 0 else len(traces)

    pe_approx_errors = torch.zeros((num_nodes,), device=device)
    pe_approx_errors_cnt = torch.zeros((num_nodes,), dtype=torch.int32, device=device)

    for trace_idx in range(num_eval_traces):
        trace = traces[trace_idx]

        with torch.no_grad():
            full_pes, last_block = compute_full_blocks(g, features, device, model, num_layers, trace, loss_fn)
            batch_size = last_block.num_dst_nodes()
            pe_nids = last_block.srcdata[dgl.NID][batch_size:]

            approx_errors = []
            for i, full_embs in enumerate(full_pes):
                pes = pe_provider.get(i, pe_nids)
                approx_errors.append(full_embs - pes)
            
            approx_errors = torch.norm(torch.stack(approx_errors), dim=-1)
            approx_errors = approx_errors.sum(dim=0)

            pe_approx_errors[pe_nids] += approx_errors
            pe_approx_errors_cnt[pe_nids] += 1
        
    used_pe_nids = pe_approx_errors_cnt > 0
    mean_pe_approx_errors = pe_approx_errors[used_pe_nids] / pe_approx_errors_cnt[used_pe_nids]

    results = {
        "mean_pe_approx_errors": mean_pe_approx_errors.to("cpu").tolist()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        f.write(json.dumps(results, indent=4, sort_keys=True))
        f.write("\n")


class PrecomputedPEProvider:
    def __init__(self, pes):
        self._pes = pes
    
    def get(self, layer_idx, nids):
        return self._pes[layer_idx][nids]

class OnlinePEProvider:
    def __init__(self, num_layers):
        self._num_layers = num_layers
        self._pes = []
        self._pe_nids = []
        self._pe_nids_sorter = []

    def add_pe(self, layer_idx, pes, nids):
        self._pes.append(pes)
        self._pe_nids.append(nids)
        self._pe_nids_sorter.append(np.argsort(nids))

    def get(self, layer_idx, nids):
        local_pe_nids = self._pe_nids_sorter[layer_idx][np.searchsorted(
            self._pe_nids[layer_idx],
            nids,
            sorter=self._pe_nids_sorter[layer_idx])]

        return self._pes[layer_idx][local_pe_nids]

def compute_full_blocks(g, features, device, model, num_layers, trace, loss_fn):
    batch_size = trace.target_gnids.shape[0]
    last_block = to_block(trace.src_gnids, trace.dst_gnids, trace.target_gnids)
    n_recomputation_targets = last_block.srcdata[dgl.NID].shape[0] - batch_size
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

    h = features[blocks[0].srcdata[dgl.NID][batch_size:]]
    h = torch.concat((trace.target_features, h)).to(device)

    full_pes = []

    with torch.no_grad():
        h0 = model.feature_preprocess(h)
        h = h0
        for layer_idx in range(num_layers):
            h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)
            if layer_idx < num_layers - 1:
                full_pes.append(h[batch_size:batch_size + n_recomputation_targets])

    return full_pes, last_block.to(device)

def create_pe_provider(g, features, device, model, num_layers, trace):
    online_pe_provider = OnlinePEProvider(num_layers)
    batch_size = trace.target_gnids.shape[0]
    block = to_block(trace.src_gnids, trace.dst_gnids, trace.target_gnids)
    blocks = [block]

    for _ in range(num_layers - 1):
        seeds = blocks[0].srcdata[dgl.NID][batch_size:]
        frontier = dgl.sampling.sample_neighbors(g, seeds, -1)
        u, v = frontier.edges()
        non_self_edge_mask = trace.dst_gnids != trace.src_gnids
        edge_mask = trace.src_gnids < trace.target_gnids.min()

        block = to_block(
            torch.concat((trace.src_gnids, u)),
            torch.concat((trace.dst_gnids, v)),
            blocks[0].srcdata[dgl.NID]
        )

        blocks.insert(0, block)

    h = features[blocks[0].srcdata[dgl.NID][batch_size:]]
    h = torch.concat((trace.target_features, h)).to(device)

    with torch.no_grad():
        h0 = model.feature_preprocess(h)
        h = h0
        for layer_idx in range(num_layers):
            h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)
            if layer_idx != num_layers - 1:
                online_pe_provider.add_pe(layer_idx, h[batch_size:].to("cpu"), blocks[layer_idx + 1].srcdata[dgl.NID][batch_size:])

        logits = h

    return online_pe_provider

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--part_config', type=str, required=True)
    parser.add_argument('--training_dir', type=str, required=True)
    parser.add_argument('--trace_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_eval_traces', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=451241)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    print(args)
    main(args)
