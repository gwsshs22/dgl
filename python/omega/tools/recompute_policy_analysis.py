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
    compute_grad = True

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

    with g.local_scope():
        g.ndata["dgr"] = 1 / g.in_degrees().clamp(min=1).type(torch.float32)
        g.update_all(fn.copy_u("dgr", "m"), fn.mean("m", "importance_score"))
        importance_score = g.ndata["importance_score"]

    if pe_data_path.exists():
        pe_data = torch.load(pe_data_path)
        pe_provider = PrecomputedPEProvider(pe_data["pes"])
    else:
        pe_provider = None

    node_feats = dgl.data.load_tensors(str(part_config_dir / "part0" / "node_feat.dgl"))
    features = node_feats["_N/features"]
    labels = node_feats["_N/labels"]

    num_eval_traces = args.num_eval_traces if args.num_eval_traces > 0 else len(traces)

    if args.thresholds is None:
        thresholds = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 60, 40, 20, 0]
    else:
        thresholds = [int(t) for t in args.thresholds.split(",")]

    full_logits_arr = []
    full_labels_arr = []
    logits_dict = defaultdict(lambda: defaultdict(list))

    if args.sampled:
        if training_config["gnn"] == "gcn":
            if num_layers == 2:
                fanouts = [10, 25]
            elif num_layers == 3:
                fanouts = [5, 10, 15]
            else:
                raise f"We do not support sampling num_layers={num_layers} for gcn model currently."
        else:
            fanouts = training_config["fanouts"]
            assert fanouts is not None
            fanouts = [int(f) for f in fanouts.split(",")]
            assert len(fanouts) == num_layers
            assert all([f > 0 for f in fanouts])
        pe_logits_arr = []

        for trace_idx in range(num_eval_traces):
            trace = traces[trace_idx]
            full_logits, full_labels = compute_full_sampled_blocks(g, features, device, model, num_layers, fanouts, trace)
            full_logits_arr.append(full_logits)
            full_labels_arr.append(full_labels)

            if pe_provider is None:
                batch_pe_provider = create_pe_provider(g, features, device, model, num_layers, trace)
            else:
                batch_pe_provider = pe_provider

            with torch.no_grad():
                logits, labels = compute_sampled_blocks_with_pe_no_recom(g, features, device, model, num_layers, fanouts, trace, batch_pe_provider)
                pe_logits_arr.append(logits)
        
        correct_labels = torch.concat(full_labels_arr)
        full_acc = get_acc_fn(correct_labels, torch.concat(full_logits_arr))
        pe_acc = get_acc_fn(correct_labels, torch.concat(pe_logits_arr))
        print(f"Full acc={full_acc}, pe acc={pe_acc}")

        results = vars(args)
        results["full_acc"] = full_acc
        results["pe_acc"] = pe_acc

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "results.json", "w") as f:
            f.write(json.dumps(results, indent=4, sort_keys=True))
            f.write("\n")
    else:
        # policy_names = ["random", "new_edge_ratios", "saint_is", "low_indegrees"]
        policy_names = ["random", "new_edge_ratios", "node_importance"]
        if compute_grad:
            policy_names.append("first_order_approx")
            policy_names.append("approx_error")
        for trace_idx in range(num_eval_traces):
            trace = traces[trace_idx]

            if compute_grad:
                full_logits, full_labels, full_pes, full_pe_grads, last_block = compute_full_blocks(g, features, device, model, num_layers, trace, loss_fn, compute_grad)
            else:
                with torch.no_grad():
                    full_logits, full_labels, full_pes, full_pe_grads, last_block = compute_full_blocks(g, features, device, model, num_layers, trace, loss_fn, compute_grad)

            full_logits_arr.append(full_logits)
            full_labels_arr.append(full_labels)

            with torch.no_grad():

                if pe_provider is None:
                    batch_pe_provider = create_pe_provider(g, features, device, model, num_layers, trace)
                else:
                    batch_pe_provider = pe_provider

                for threshold in thresholds:
                    logits, _ = compute_with_recomputation(g, features, device, model, num_layers, trace, batch_pe_provider, random_policy, threshold)
                    logits_dict["random"][threshold].append(logits)

                    logits, _ = compute_with_recomputation(g, features, device, model, num_layers, trace, batch_pe_provider, new_edge_ratios_policy, threshold)
                    logits_dict["new_edge_ratios"][threshold].append(logits)

                    def node_importance_policy(trace, last_block, new_in_degrees, org_in_degrees):
                        batch_size = last_block.num_dst_nodes()
                        pe_nids = last_block.srcdata[dgl.NID][batch_size:]
                        return importance_score[pe_nids]

                    if "node_importance" in policy_names:
                        logits, _ = compute_with_recomputation(g, features, device, model, num_layers, trace, batch_pe_provider, node_importance_policy, threshold)
                        logits_dict["node_importance"][threshold].append(logits)

                    if "low_indegrees" in policy_names:
                        logits, _ = compute_with_recomputation(g, features, device, model, num_layers, trace, batch_pe_provider, low_indegrees_policy, threshold)
                        logits_dict["low_indegrees"][threshold].append(logits)

                    if "saint_is" in policy_names:
                        logits, _ = compute_with_recomputation(g, features, device, model, num_layers, trace, batch_pe_provider, saint_is_policy, threshold)
                        logits_dict["saint_is"][threshold].append(logits)

                    if compute_grad:
                        def first_order_approx_policy(trace, last_block, new_in_degrees, org_in_degrees):
                            batch_size = last_block.num_dst_nodes()
                            pe_nids = last_block.srcdata[dgl.NID][batch_size:]

                            return torch.norm((full_pes[-1] - batch_pe_provider.get(num_layers - 2, pe_nids)) * full_pe_grads[batch_size:], dim=-1)

                        def approx_error_policy(trace, last_block, new_in_degrees, org_in_degrees):
                            batch_size = last_block.num_dst_nodes()
                            pe_nids = last_block.srcdata[dgl.NID][batch_size:]
                            n_recomputation_targets = pe_nids.shape[0]

                            scores = None
                            for l in range(num_layers - 1):
                                if scores is None:
                                    scores = torch.norm((full_pes[l][:n_recomputation_targets] - batch_pe_provider.get(l, pe_nids)), dim=-1)
                                else:
                                    scores += torch.norm((full_pes[l][:n_recomputation_targets] - batch_pe_provider.get(num_layers - 2, pe_nids)), dim=-1)


                            return scores


                        logits, _ =  compute_with_recomputation(g, features, device, model, num_layers, trace, batch_pe_provider, first_order_approx_policy, threshold)
                        logits_dict["first_order_approx"][threshold].append(logits)
                        
                        logits, _ =  compute_with_recomputation(g, features, device, model, num_layers, trace, batch_pe_provider, approx_error_policy, threshold)
                        logits_dict["approx_error"][threshold].append(logits)

        policy_accs = defaultdict(list)

        labels = torch.concat(full_labels_arr)

        full_acc = get_acc_fn(labels, torch.concat(full_logits_arr))

        print(f"Full acc={full_acc}")

        for policy_name in policy_names: 
            for threshold in thresholds:
                acc = get_acc_fn(labels, torch.concat(logits_dict[policy_name][threshold]))
                policy_accs[policy_name].append(acc)
        results = vars(args)
        results["thresholds"] = thresholds
        results["policy_accs"] = policy_accs
        results["full_acc"] = full_acc

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "results.json", "w") as f:
            f.write(json.dumps(results, indent=4, sort_keys=True))
            f.write("\n")

        save_figure(output_dir, training_config, thresholds, full_acc, policy_accs)


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

def new_edge_ratios_policy(trace, last_block, new_in_degrees, org_in_degrees):
    return (new_in_degrees) / (new_in_degrees + org_in_degrees)

def low_indegrees_policy(trace, last_block, new_in_degrees, org_in_degrees):
    return 1 / org_in_degrees

def saint_is_policy(trace, last_block, new_in_degrees, org_in_degrees):
    batch_size = last_block.num_dst_nodes()
    pe_nids = last_block.srcdata[dgl.NID][batch_size:]
    min_target_id = trace.target_gnids.min()

    edge_mask = torch.logical_and(
        trace.dst_gnids >= min_target_id,
        trace.src_gnids < min_target_id)

    block = to_block(trace.dst_gnids[edge_mask], trace.src_gnids[edge_mask], pe_nids, trace.target_gnids)

    block.srcdata["s"] = 1/block.out_degrees()
    block.dstdata["d"] = 1/(new_in_degrees + org_in_degrees)
    block.apply_edges(fn.u_add_v("s", "d", "h"))
    block.update_all(fn.copy_e("h","m"), fn.sum("m", "r"))
    return block.dstdata["r"]

def random_policy(trace, last_block, new_in_degrees, org_in_degrees):
    return torch.rand(last_block.num_src_nodes() - last_block.num_dst_nodes())

def compute_with_recomputation(g, features, device, model, num_layers, trace, pe_provider, policy, threshold):
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

def compute_full_blocks(g, features, device, model, num_layers, trace, loss_fn, compute_grad):
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
    if compute_grad:
        with torch.no_grad():
            h0 = model.feature_preprocess(h)
            h = h0
            for layer_idx in range(num_layers - 1):
                h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)
                full_pes.append(h[batch_size:batch_size + n_recomputation_targets].to("cpu"))
        h.requires_grad_(True)
        h.retain_grad()
        logits = model.layer_foward(num_layers - 1, blocks[num_layers - 1].to(device), h, h0)
        
        loss = loss_fn(logits, trace.target_labels.to(device))
        loss.backward()

        pe_grads = h.grad

        return logits.detach().to("cpu"), trace.target_labels, full_pes, pe_grads.to("cpu"), last_block
    else:
        with torch.no_grad():
            h0 = model.feature_preprocess(h)
            h = h0
            for layer_idx in range(num_layers):
                h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)
                if layer_idx < num_layers - 1:
                    full_pes.append(h[batch_size:batch_size + n_recomputation_targets].to("cpu"))

            logits = h
        return logits.to("cpu"), trace.target_labels, full_pes, None, last_block

def compute_full_sampled_blocks(g, features, device, model, num_layers, fanouts, trace):
    sample_ret = sample_edges(trace.target_gnids, trace.src_gnids, trace.dst_gnids, fanouts)

    src_gnids = sample_ret[0][0]
    dst_gnids = sample_ret[0][1]

    batch_size = trace.target_gnids.shape[0]
    last_block = to_block(src_gnids, dst_gnids, trace.target_gnids)

    blocks = [last_block]

    for layer_idx in range(1, num_layers):
        fanout = fanouts[layer_idx]

        seeds = blocks[0].srcdata[dgl.NID][batch_size:]
        f = dgl.sampling.sample_neighbors(g, seeds, fanout)

        u, v = f.edges()

        block = to_block(
            torch.concat((sample_ret[layer_idx][0], u)),
            torch.concat((sample_ret[layer_idx][1], v)),
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

    with torch.no_grad():
        h0 = model.feature_preprocess(h)
        h = h0
        for layer_idx in range(num_layers):
            h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)

        logits = h
    return logits.to("cpu"), trace.target_labels

def compute_sampled_blocks_with_pe_no_recom(g, features, device, model, num_layers, fanouts, trace, pe_provider):
    batch_size = trace.target_gnids.shape[0]
    sample_ret = sample_edges(trace.target_gnids, trace.src_gnids, trace.dst_gnids, fanouts)
    blocks = []
    for u, v in sample_ret:
        block = to_block(
            u,
            v,
            trace.target_gnids
        )
        blocks.insert(0, block)

    with torch.no_grad():
        features = torch.concat((
            trace.target_features.to(device),
            features[blocks[0].srcdata[dgl.NID][batch_size:]].to(device)
        ))

        h0 = model.feature_preprocess(features)

        h = h0
        h = model.layer_foward(0, blocks[0].to(device), h, h0)

        for layer_idx in range(1, num_layers):
            h = torch.concat((h, pe_provider.get(layer_idx - 1, blocks[layer_idx].srcdata[dgl.NID][batch_size:]).to(device)))
            h = model.layer_foward(layer_idx, blocks[layer_idx].to(device), h, h0)

        return h.cpu(), trace.target_labels

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

def save_figure(output_dir, training_config, thresholds, full_acc, policy_accs):
    title = f"{training_config['graph_name']}, {training_config['num_layers']}-Layer {training_config['gnn'].upper()}"

    thresholds = np.array(thresholds)
    acc_values = { k: np.array(v) for k, v in policy_accs.items() }

    plt.figure(figsize=(8, 6))

    x_arr = 100.0 - thresholds

    for name, acc_arr in acc_values.items():
        plt.plot(x_arr, (full_acc - acc_arr) * 100, label=name)

    plt.legend()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # plt.ylim([0, None])

    plt.axhline(y=1.0, linestyle='dotted', color='red')

    plt.ylabel('Accuracy Drop (%)', fontsize=20)
    plt.xlabel('PE Recomputation (%)', fontsize=20)
    plt.title(title, fontsize=30)

    filename = f"{training_config['graph_name']}_{training_config['gnn'].lower()}_{training_config['num_layers']}.png"
    plt.savefig(str(output_dir / filename), bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--part_config', type=str, required=True)
    parser.add_argument('--training_dir', type=str, required=True)
    parser.add_argument('--trace_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--thresholds', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--num_eval_traces', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=451241)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    print(args)
    main(args)
