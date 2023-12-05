import argparse
import os
import json
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt

import dgl
from dgl import function as fn

from dgl.omega.omega_apis import to_block
from omega.utils import get_dataset_config
from omega.utils import load_traces
from omega.utils import cal_metrics
from omega.models import load_model_from

def compute_with_recomputation(g, features, device, model, num_layers, trace, pe_provider, policy, threshold, gcn_both_norm):
    batch_size = trace.target_gnids.shape[0]
    block = to_block(trace.src_gnids, trace.dst_gnids, trace.target_gnids)

    tmp_block = to_block(
        trace.dst_gnids,
        trace.src_gnids,
        block.srcdata[dgl.NID]
    )

    new_in_degrees = tmp_block.in_degrees()[batch_size:]
    org_in_degrees = g.in_degrees(block.srcdata[dgl.NID][batch_size:])

    if threshold == 100:
        recompute_mask = torch.zeros(block.num_src_nodes() - batch_size, dtype=torch.bool)
    else:
        metrics = policy(
            trace,
            block,
            new_in_degrees,
            org_in_degrees)

        m_threshold = np.percentile(metrics, threshold)
        recompute_mask = metrics >= m_threshold

    recompute_ids = block.srcdata[dgl.NID][batch_size:][recompute_mask]
    reuse_ids = block.srcdata[dgl.NID][batch_size:][torch.logical_not(recompute_mask)]

    frontier = dgl.sampling.sample_neighbors(g, recompute_ids, -1)
    u, v = frontier.edges()

    edge_mask = trace.src_gnids < trace.target_gnids.min()
    tmp_g = dgl.graph((
        torch.concat((trace.src_gnids, trace.dst_gnids[edge_mask])),
        torch.concat((trace.dst_gnids, trace.src_gnids[edge_mask]))
    ))

    recompute_block_seeds = torch.concat((trace.target_gnids, recompute_ids))
    u2, v2 = tmp_g.in_edges(recompute_block_seeds, 'uv')

    recompute_block = to_block(
        torch.concat((u2, u)),
        torch.concat((v2, v)),
        recompute_block_seeds
    )

    h = torch.concat((
        trace.target_features,
        features[recompute_block.srcdata[dgl.NID][batch_size:]]))

    h = h.to(device)

    if gcn_both_norm:
        block.set_out_degrees(torch.concat((
            tmp_block.out_degrees()[:batch_size],
            g.out_degrees(block.srcdata[dgl.NID][batch_size:]))
        ))
        recompute_block.set_out_degrees(torch.concat((
            tmp_block.out_degrees()[:batch_size],
            g.out_degrees(recompute_block.srcdata[dgl.NID][batch_size:]))
        ))

    recompute_block = recompute_block.to(device)
    block = block.to(device)

    h0 = model.feature_preprocess(h)
    h = h0    
    h = model.layer_foward(0, recompute_block, h, h0)

    for layer_idx in range(1, num_layers - 1):
        p = pe_provider.get(layer_idx - 1, recompute_block.srcdata[dgl.NID][h.shape[0]:].to("cpu")).to(device)
        h = torch.concat((h, p))
        h = model.layer_foward(layer_idx, recompute_block, h, h0)

    pe = torch.zeros((block.num_src_nodes() - block.num_dst_nodes(),) + h.shape[1:], device=device)
    pe[recompute_mask] = h[batch_size:]
    pe[torch.logical_not(recompute_mask)] = p = pe_provider.get(num_layers - 2, reuse_ids).to(device)
    h = torch.concat((
        h[:batch_size],
        pe
    ))

    h = model.layer_foward(num_layers - 1, block, h, h0)

    return h.cpu(), trace.target_labels

def compute_full_blocks(g, features, device, model, num_layers, trace, loss_fn, compute_grad, gcn_both_norm):
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

    h = features[blocks[0].srcdata[dgl.NID][batch_size:]]
    h = torch.concat((trace.target_features, h)).to(device)

    if compute_grad:
        with torch.no_grad():
            h0 = model.feature_preprocess(h)
            h = h0
            for layer_idx in range(num_layers - 1):
                h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)
        h.requires_grad_(True)
        h.retain_grad()
        logits = model.layer_foward(num_layers - 1, blocks[num_layers - 1].to(device), h, h0)
        
        loss = loss_fn(logits, trace.target_labels.to(device))
        loss.backward()

        pe_grads = h.grad

        return logits.detach().to("cpu"), trace.target_labels, h.detach().to("cpu"), pe_grads.to("cpu"), last_block
    else:
        with torch.no_grad():
            h0 = model.feature_preprocess(h)
            h = h0
            for layer_idx in range(num_layers):
                if layer_idx == num_layers - 1:
                    exact_pes = h
                h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)

            logits = h
        return logits.to("cpu"), trace.target_labels, exact_pes, None, last_block

class PrecomputedPEProvider:
    def __init__(self, pes):
        self._pes = pes
    
    def get(self, layer_idx, nids):
        return self._pes[layer_idx][nids]

def get_large_error_degrees(g, features, training_dir, device, traces, part_config_dir, graph_name, threshold):
    model, training_config, dataset_config = load_model_from(training_dir)
    training_id = training_config["id"]
    num_layers = training_config["num_layers"]
    assert training_config["graph_name"] == graph_name

    pe_data = torch.load(part_config_dir / "part0" / "data" / training_id / "tensors.pth")
    pes = pe_data["pes"]
    pe_provider = PrecomputedPEProvider(pes)
    global_recompute_nids = []
    global_pe_nids = []
    global_foes = []
    compute_grad = True

    if dataset_config.multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    model = model.to(device)
    model.eval()
    gcn_both_norm = (training_config["gnn"] == "gcn" or training_config["gnn"] == "gcn2") and training_config["gcn_norm"] == "both"

    for trace in traces:
        full_logits, full_labels, full_pes, full_pe_grads, last_block = compute_full_blocks(
            g,
            features,
            device,
            model,
            num_layers,
            trace,
            loss_fn,
            compute_grad,
            gcn_both_norm)

        def gradient_policy(trace, last_block, new_in_degrees, org_in_degrees):
            batch_size = last_block.num_dst_nodes()
            pe_nids = last_block.srcdata[dgl.NID][batch_size:]

            metrics = torch.norm((full_pes[batch_size:] - pes[-1][pe_nids]) * full_pe_grads[batch_size:], dim=-1)
            global_foes.append(metrics)
            m_th = np.percentile(metrics, threshold)
            global_recompute_nids.append(
                pe_nids[metrics >= m_th]
            )
            global_pe_nids.append(pe_nids)
            return metrics

        with torch.no_grad():
            logits, _ = compute_with_recomputation(g, features, device, model, num_layers, trace, pe_provider, gradient_policy, threshold, gcn_both_norm)

    return torch.concat(global_pe_nids), torch.concat(global_recompute_nids), torch.concat(global_foes)


def get_gat_attn_nids(g, features, training_dir, device, traces, graph_name, threshold):
    model, training_config, dataset_config = load_model_from(training_dir)
    training_id = training_config["id"]
    num_layers = training_config["num_layers"]
    assert training_config["graph_name"] == graph_name
    assert training_config["gnn"] == "gat"
    
    model = model.to(device)
    model.eval()

    global_gat_attn_nids = []

    for trace in traces:

        def compute_full_blocks_gat(g, features, device, model, num_layers, trace):
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
                    torch.concat((trace.src_gnids, trace.dst_gnids[edge_mask], u)),
                    torch.concat((trace.dst_gnids, trace.src_gnids[edge_mask], v)),
                    blocks[0].srcdata[dgl.NID]
                )

                blocks.insert(0, block)

            h = features[blocks[0].srcdata[dgl.NID][batch_size:]]
            h = torch.concat((trace.target_features, h)).to(device)

            with torch.no_grad():
                h0 = model.feature_preprocess(h)
                h = h0

                for layer_idx in range(num_layers):
                    if layer_idx == num_layers - 1:
                        exact_pes = h
                    h, attn = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0, get_attention=True)

                first_block = blocks[-1]

                norm_attn_values = defaultdict(int)
                norm_attn_cnts = defaultdict(int)
                u_arr, v_arr = first_block.edges()
                u_arr = first_block.srcdata[dgl.NID][u_arr]
                v_arr = first_block.srcdata[dgl.NID][v_arr]

                for u, v, a in zip(u_arr.tolist(), v_arr.tolist(), attn.squeeze().mean(dim=-1).to("cpu").tolist()):
                    norm_attn_values[u] += a
                    norm_attn_cnts[u] += 1

                norm_attn = []
                pe_nids = blocks[-1].srcdata[dgl.NID][batch_size:]
                for pe_nid in pe_nids.tolist():
                    norm_attn.append(norm_attn_values[pe_nid] / norm_attn_cnts[pe_nid])


                logits = h
            return logits.to("cpu"), trace.target_labels, exact_pes, None, torch.tensor(norm_attn), pe_nids

        _, _, _, _, norm_attn, pe_nids = compute_full_blocks_gat(
                g,
                features,
                device,
                model,
                num_layers,
                trace)

        norm_th = np.percentile(norm_attn, threshold)
        gat_attn_nids = pe_nids[norm_attn >= norm_th]
        global_gat_attn_nids.append(gat_attn_nids)

    
    return torch.concat(global_gat_attn_nids)

def main(args):
    graph_name = args.graph_name
    analysis_root = args.analysis_root
    gat_training_dir = args.gat_training_dir
    gcn_training_dir = args.gcn_training_dir
    local_rank = args.local_rank
    threshold = args.threshold

    part_config = f"{analysis_root}/datasets/{graph_name}/{graph_name}.json"
    trace_dir = f"{analysis_root}/traces/{graph_name}-random-1024"
    local_rank = args.local_rank

    compute_grad = True

    device = f"cuda:{local_rank}"

    part_config_path = Path(part_config)
    part_config = json.loads(part_config_path.read_text())
    part_config_dir = part_config_path.parent

    traces = load_traces(trace_dir)

    assert part_config["num_parts"] == 1

    g = dgl.load_graphs(str(part_config_dir / "part0" / "graph.dgl"))[0][0]

    node_feats = dgl.data.load_tensors(str(part_config_dir / "part0" / "node_feat.dgl"))
    features = node_feats["_N/features"]
    labels = node_feats["_N/labels"]

    gat_all_pe_nids, gat_error_pe_nids, gat_foes = get_large_error_degrees(g, features, gat_training_dir, device, traces, part_config_dir, graph_name, threshold)
    gcn_all_pe_nids, gcn_error_pe_nids, gcn_foes = get_large_error_degrees(g, features, gcn_training_dir, device, traces, part_config_dir, graph_name, threshold)
    gat_attn_pe_nids = get_gat_attn_nids(g, features, gat_training_dir, device, traces, graph_name, threshold)

    all_degrees = g.in_degrees(gat_all_pe_nids)
    gat_error_degrees = g.in_degrees(gat_error_pe_nids)
    gcn_error_degrees = g.in_degrees(gcn_error_pe_nids)
    gat_attn_degrees = g.in_degrees(gat_attn_pe_nids)

    args_dict = vars(args)
    args_dict["all_pe_nids"] = gat_all_pe_nids.tolist()
    args_dict["gat_error_pe_nids"] = gat_error_pe_nids.tolist()
    args_dict["gcn_all_pe_nids"] = gcn_all_pe_nids.tolist()
    args_dict["gat_attn_pe_nids"] = gat_attn_pe_nids.tolist()

    args_dict["all_degrees"] = all_degrees.tolist()
    args_dict["gat_error_degrees"] = gat_error_degrees.tolist()
    args_dict["gcn_error_degrees"] = gcn_error_degrees.tolist()
    args_dict["gat_attn_degrees"] = gat_attn_degrees.tolist()

    args_dict["gat_foes"] = gat_foes.tolist()
    args_dict["gcn_foes"] = gcn_foes.tolist()

    result_file_path = Path(args.result_file_path)
    os.makedirs(str(result_file_path.parent), exist_ok=True)
    result_file_path.write_text(json.dumps(args_dict))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--result_file_path', type=str, required=True)
    parser.add_argument('--analysis_root', type=str, required=True)
    parser.add_argument('--gat_training_dir', type=str, required=True)
    parser.add_argument('--gcn_training_dir', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=90)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main(args)
