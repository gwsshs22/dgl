from pathlib import Path

import dgl
import numpy as np
import torch as th
import argparse
import time
import sys
import os
import gc
import json

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .load_graph import load_reddit, load_ogb
from .dist_partition import partition_graph

def save_infer_graph(args, org_g):
    output_path = Path(args.output)
    os.makedirs(output_path, exist_ok=True)
    infer_prob = args.infer_prob
    # Save infer_target_graph
    print("Build inference target graph.")

    # Build infer_target_g by selecting about 10% of nodes in the graph
    infer_target_mask = th.BoolTensor(np.random.choice([False, True], size=(org_g.number_of_nodes(),), p=[1 - infer_prob, infer_prob]))
    num_infer_targets = infer_target_mask.sum()
    org_g.ndata['infer_target_mask'] = infer_target_mask

    infer_target_nids = th.masked_select(th.arange(org_g.number_of_nodes()), infer_target_mask)
    infer_target_in_eids = org_g.in_edges(infer_target_nids, 'eid')
    infer_target_out_eids = org_g.out_edges(infer_target_nids, 'eid')
    infer_target_eids = np.union1d(infer_target_in_eids, infer_target_out_eids)
    infer_target_g = org_g.edge_subgraph(infer_target_eids)

    dgl.save_graphs(str(output_path / "infer_target_graph.dgl"), infer_target_g)

    id_mappings_path = str(Path(args.output) / "id_mappings.dgl")
    if os.path.exists(id_mappings_path):
        id_mappings = dgl.data.load_tensors(id_mappings_path)
    else:
        id_mappings = {}
    id_mappings["infer_target_mask"] = infer_target_mask
    id_mappings["infer_target_orig_ids"] = infer_target_nids
    dgl.data.save_tensors(id_mappings_path, id_mappings)

    print("Build inference target graph done.")

def partition(args, org_g):
    output_path = Path(args.output)
    os.makedirs(output_path, exist_ok=True)

    # Partition the origin graph
    partition_graph(org_g, args.dataset, args.num_parts, args.output,
                    part_method=args.part_method,
                    balance_ntypes=balance_ntypes,
                    balance_edges=args.balance_edges,
                    num_trainers_per_machine=args.num_trainers_per_machine,
                    target_part=args.target_part)


def finalize(args, org_g):
    merge_config_files(args)
    collect_id_mappings(args)
    fill_node_features(args, org_g)
    save_infer_target_features(args, org_g)
    delete_infer_edges(args, org_g)

def merge_config_files(args):
    if os.path.exists(Path(args.output) / f'{args.dataset}.json'):
        return

    merged_meta = None
    for part_id in range(args.num_parts):
        part_config = Path(args.output) / f'{args.dataset}-{part_id}.json'
        with open(part_config) as conf_f:
            part_metadata = json.load(conf_f)
        if part_id == 0:
            merged_meta = part_metadata
            continue
        
        merged_meta[f"part-{part_id}"] = part_metadata[f"part-{part_id}"]
        merged_meta["node_map"]['_N'].append(part_metadata["node_map"]['_N'][0])
    
    with open(Path(args.output) / f'{args.dataset}.json', 'w') as outfile:
        json.dump(merged_meta, outfile, sort_keys=True, indent=4)


def collect_id_mappings(args):
    # Collect id mappings to build inference requests later
    orig_ids_in_partitions = []
    global_ids_in_partitions = []
    for part_id in range(args.num_parts):
        part_graph_path = Path(args.output) / f'part{part_id}' / 'graph.dgl'
        part_graph = dgl.load_graphs(str(part_graph_path))[0][0]

        num_inner_nodes = part_graph.ndata["inner_node"].sum().item()
        orig_ids_in_partitions.append(part_graph.ndata["orig_id"][:num_inner_nodes])
        global_ids_in_partitions.append(part_graph.ndata[dgl.NID][:num_inner_nodes])

    id_mappings_path = str(Path(args.output) / "id_mappings.dgl")
    if os.path.exists(id_mappings_path):
        id_mappings = dgl.data.load_tensors(id_mappings_path)
    else:
        id_mappings = {}
    id_mappings["orig_ids_in_partitions"] = th.concat(orig_ids_in_partitions)
    id_mappings["global_ids_in_partitions"] = th.concat(global_ids_in_partitions)
    dgl.data.save_tensors(str(Path(args.output) / "id_mappings.dgl"), id_mappings)


def fill_node_features(args, org_g):
    id_mappings = dgl.data.load_tensors(str(Path(args.output) / "id_mappings.dgl"))

    node_features = org_g.ndata['features']
    if args.for_training:
        labels = org_g.ndata['labels']
    for part_id in range(args.num_parts):
        part_graph_path = Path(args.output) / f'part{part_id}' / 'graph.dgl'
        part_graph = dgl.load_graphs(str(part_graph_path))[0][0]

        num_inner_nodes = part_graph.ndata["inner_node"].sum().item()
        orig_ids = part_graph.ndata["orig_id"][:num_inner_nodes]

        part_node_features = node_features[orig_ids]

        if args.for_training:
            infer_target_mask = id_mappings["infer_target_mask"][orig_ids]
            train_mask = torch.logical_not(infer_target_mask)
            node_feats = { "_N/features": part_node_features, "_N/labels": labels[orig_ids], "_N/infer_target_mask": infer_target_mask, "_N/train_mask": train_mask }
        else:
            node_feats = { "_N/features": part_node_features }
        dgl.data.save_tensors(str(Path(args.output) / f"part{part_id}" / "node_feat.dgl"), node_feats)

def save_infer_target_features(args, org_g):
    output_path = Path(args.output)
    node_features = org_g.ndata['features']

    id_mappings = dgl.data.load_tensors(str(Path(args.output) / "id_mappings.dgl"))
    infer_target_orig_ids = id_mappings["infer_target_orig_ids"]
    infer_target_features = node_features[infer_target_orig_ids]
    id_mappings["infer_target_features"] = infer_target_features
    dgl.data.save_tensors(str(Path(args.output) / "id_mappings.dgl"), id_mappings)

def delete_infer_edges(args, org_g):
    output_path = Path(args.output)
    id_mappings = dgl.data.load_tensors(str(Path(args.output) / "id_mappings.dgl"))
    
    infer_target_mask = id_mappings["infer_target_mask"]
    infer_target_orig_ids = id_mappings["infer_target_orig_ids"]
    for part_id in range(args.num_parts):
        part_graph_path = Path(args.output) / f'part{part_id}' / 'graph.dgl'
        part_graph = dgl.load_graphs(str(part_graph_path))[0][0]

        part_orig_id = part_graph.ndata["orig_id"]
        part_graph.ndata["infer_target_mask"] = infer_target_mask[part_orig_id]
        if args.for_training:
            part_graph.ndata["train_mask"] = torch.logical_not(part_graph.ndata["infer_target_mask"])
        infer_targets_orig_ids_in_part = np.intersect1d(infer_target_orig_ids, part_orig_id)

        sorter = np.argsort(part_orig_id)
        infer_targets_ids_in_part = sorter[np.searchsorted(part_orig_id, infer_targets_orig_ids_in_part, sorter=sorter)]
        
        infer_in_edge_eids = part_graph.in_edges(infer_targets_ids_in_part, 'eid')
        infer_out_edge_eids = part_graph.out_edges(infer_targets_ids_in_part, 'eid')

        part_graph.remove_edges(np.union1d(infer_in_edge_eids, infer_out_edge_eids))
        dgl.save_graphs(str(part_graph_path), [part_graph])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogbn-products, ogbn-papers100M')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    argparser.add_argument('--random_seed', type=int, default=42421)
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.')
    argparser.add_argument('--infer_prob', type=float, default=0.1)
    argparser.add_argument('--target_part', type=int, default=-1)
    argparser.add_argument('--for_training', action="store_true")
    argparser.add_argument('--stage', type=str, default="all", choices=["save_infer_graph", "partition", "finalize"])
    args = argparser.parse_args()
    np.random.seed(args.random_seed)
    start = time.time()
    if args.dataset == 'reddit':
        g, _ = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-arxiv':
        g, _ = load_ogb('ogbn-arxiv')
    elif args.dataset == 'ogbn-papers100M':
        g, _ = load_ogb('ogbn-papers100M')
    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))

    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    node_feature_keys = set(g.ndata.keys())
    for k in node_feature_keys:
        if k == "features":
            continue

        if args.for_training and k == 'labels':
            continue

        del g.ndata[k]

    if args.stage == "save_infer_graph":
        save_infer_graph(args, g)
    elif args.stage == "partition":
        del g.ndata["features"]
        partition(args, g)
    elif args.stage == "finalize":
        finalize(args, g)
