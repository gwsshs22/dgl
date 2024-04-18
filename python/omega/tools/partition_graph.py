import argparse
import time
import sys
import os
import gc
import json
from pathlib import Path

import dgl
import numpy as np
import torch as th

from omega.utils import load_graph

def save_infer_graph(args, org_g):
    output_path = Path(args.output)
    os.makedirs(output_path, exist_ok=True)
    infer_prob = args.infer_prob

    # Save infer_target_graph
    print("Build inference target graph.")

    # Build infer_target_g by selecting about 10% of test nodes in the graph
    all_nids = th.arange(org_g.number_of_nodes())
    test_nids = all_nids[org_g.ndata["test_mask"]]

    if args.rel_to_tests:
        num_infer_targets = int(infer_prob * test_nids.shape[0])
    else:
        num_infer_targets = min(int(infer_prob * org_g.number_of_nodes()), test_nids.shape[0])

    infer_prob_within_test_nodes = num_infer_targets / test_nids.shape[0]
    mask = th.BoolTensor(np.random.choice(
        [False, True],
        size=(test_nids.shape[0],),
        p=[1 - infer_prob_within_test_nodes, infer_prob_within_test_nodes]
    ))

    infer_target_nids = test_nids[mask]
    infer_target_mask = th.zeros(org_g.number_of_nodes(), dtype=th.bool)
    infer_target_mask[infer_target_nids] = True

    num_infer_targets = infer_target_mask.sum()
    print(f"infer_prob={infer_prob}, num_infer_targets/num_nodes = {num_infer_targets / org_g.number_of_nodes()}")
    org_g.ndata['infer_target_mask'] = infer_target_mask

    id_arr = th.arange(org_g.number_of_nodes())

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

    dgl.data.save_tensors(id_mappings_path, id_mappings)

    print("Build inference target graph done.")

def partition_org_graph(args, org_g):
    orig_nids, _ = dgl.distributed.partition_graph(
        org_g, args.dataset, args.num_parts, args.output,
        part_method=args.part_method,
        include_out_edges=args.include_out_edges,
        num_hops=1,
        return_mapping=True)
    id_mappings_path = str(Path(args.output) / "id_mappings.dgl")
    id_mappings = dgl.data.load_tensors(id_mappings_path)
    id_mappings["orig_nids"] = orig_nids
    dgl.data.save_tensors(id_mappings_path, id_mappings)

def delete_infer_edges(args):
    output_path = Path(args.output)
    id_mappings = dgl.data.load_tensors(str(Path(args.output) / "id_mappings.dgl"))

    infer_target_mask = id_mappings["infer_target_mask"]
    orig_nids = id_mappings["orig_nids"]

    inner_edge_counts = []
    for part_id in range(args.num_parts):
        part_graph_path = Path(args.output) / f'part{part_id}' / 'graph.dgl'
        part_graph = dgl.load_graphs(str(part_graph_path))[0][0]

        part_infer_target_mask = infer_target_mask[orig_nids[part_graph.ndata[dgl.NID]]]
        part_infer_target_local_ids = th.masked_select(th.arange(part_infer_target_mask.shape[0]), part_infer_target_mask)
        infer_in_edge_eids = part_graph.in_edges(part_infer_target_local_ids, 'eid')
        infer_out_edge_eids = part_graph.out_edges(part_infer_target_local_ids, 'eid')
        part_graph.remove_edges(np.union1d(infer_in_edge_eids, infer_out_edge_eids))
        inner_edge_counts.append(part_graph.edata["inner_edge"].sum().item())
        dgl.save_graphs(str(part_graph_path), [part_graph])

    config = json.loads((Path(args.output) / f'{args.dataset}.json').read_text())
    l = []
    start_idx = 0
    for inner_edge_count in inner_edge_counts:
        l.append([start_idx, start_idx + inner_edge_count])
        start_idx += inner_edge_count

    assert "_N:_E:_N" in config["edge_map"]
    config["edge_map"]["_N:_E:_N"] = l
    config["num_edges"] = l[-1][-1]

    with open(Path(args.output) / f'{args.dataset}.json', 'w') as outfile:
        json.dump(config, outfile, sort_keys=True, indent=4)

def main(args):
    org_g = load_graph(args.dataset, ogbn_data_root=args.ogbn_data_root, saint_data_root=args.saint_data_root, igb_data_root=args.igb_data_root)
    save_infer_graph(args, org_g)
    partition_org_graph(args, org_g)
    del org_g
    gc.collect()
    delete_infer_edges(args)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogbn-products, ogbn-papers100M, amazon, yelp, flickr, igb-tiny')
    argparser.add_argument('--ogbn_data_root', type=str)
    argparser.add_argument('--saint_data_root', type=str)
    argparser.add_argument('--igb_data_root', type=str)
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='random', choices=["random", "metis"],
                           help='the partition method')
    argparser.add_argument('--include_out_edges', action='store_true')
    argparser.add_argument('--random_seed', type=int, default=412523)
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.', required=True)
    argparser.add_argument('--infer_prob', type=float, default=0.1)
    argparser.add_argument('--rel_to_tests', action="store_true")

    args = argparser.parse_args()
    np.random.seed(args.random_seed)

    if args.dataset == "amazon" or args.dataset == "yelp" or args.dataset == "flickr":
        assert args.saint_data_root is not None
    elif args.dataset == "ogbn-products" or args.dataset == "ogbn-papers100M":
        assert args.ogbn_data_root is not None

    main(args)
