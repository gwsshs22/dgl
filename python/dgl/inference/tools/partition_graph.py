from pathlib import Path

import dgl
import numpy as np
import torch as th
import argparse
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from load_graph import load_reddit, load_ogb

def remove_isolated_nodes(g):
    in_degrees = g.in_degrees()
    out_degrees = g.out_degrees()

    filter_mask = (in_degrees + out_degrees) == 0
    print(f"Filter {filter_mask.sum()} isolated nodes")
    if filter_mask.sum() == 0:
      return g

    non_isolated_nids = th.masked_select(th.arange(g.number_of_nodes()), th.logical_not(filter_mask))
    return g.subgraph(non_isolated_nids)

def partition_graph(args, old_g):
    output_path = Path(args.output)

    # Remove isolated nodes
    old_g = remove_isolated_nodes(old_g)

    # Build infer_target_g by selecting about 10% of nodes in the graph
    infer_target_mask = th.BoolTensor(np.random.choice([False, True], size=(old_g.number_of_nodes(),), p=[0.9, 0.1]))
    num_infer_targets = infer_target_mask.sum()
    old_g.ndata['infer_target_mask'] = infer_target_mask

    infer_target_nids = th.masked_select(th.arange(old_g.number_of_nodes()), infer_target_mask)
    infer_target_in_eids = old_g.in_edges(infer_target_nids, 'eid')
    infer_target_out_eids = old_g.out_edges(infer_target_nids, 'eid')
    infer_target_eids = np.union1d(infer_target_in_eids, infer_target_out_eids)
    infer_target_g = old_g.edge_subgraph(infer_target_eids)

    # Save infer_target_graph
    os.makedirs(output_path, exist_ok=True)
    dgl.save_graphs(str(output_path / "infer_target_graph.dgl"), infer_target_g)
    del infer_target_g

    # Make a new graph that does not contain any inference targets
    new_g = old_g.subgraph(th.logical_not(infer_target_mask))
    del old_g

    if args.save_new_g:
        dgl.save_graphs(str(output_path / "new_graph.dgl"), new_g)

    id_mappings = {
        "new_to_old": new_g.ndata[dgl.NID]
    }

    # Partition the new graph
    dgl.distributed.partition_graph(new_g, args.dataset, args.num_parts, args.output,
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges,
                                    num_trainers_per_machine=args.num_trainers_per_machine)
    del new_g

    # Collect id mappings to build inference requests later
    new_ids_in_partitions = []
    global_ids_in_partitions = []
    for part_id in range(args.num_parts):
        part_graph_path = Path(args.output) / f'part{part_id}' / 'graph.dgl'
        part_graph = dgl.load_graphs(str(part_graph_path))[0][0]

        num_inner_nodes = part_graph.ndata["inner_node"].sum().item()
        new_ids_in_partitions.append(part_graph.ndata["orig_id"][:num_inner_nodes])
        global_ids_in_partitions.append(part_graph.ndata[dgl.NID][:num_inner_nodes])

    id_mappings["new_ids_in_partitions"] = th.concat(new_ids_in_partitions)
    id_mappings["global_ids_in_partitions"] = th.concat(global_ids_in_partitions)

    dgl.data.save_tensors(str(Path(args.output) / "id_mappings.dgl"), id_mappings)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogbn-products, ogbn-paper100M')
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
    argparser.add_argument('--save_new_g', action='store_true')
    args = argparser.parse_args()
    np.random.seed(args.random_seed)
    start = time.time()
    if args.dataset == 'reddit':
        g, _ = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-arxiv':
        g, _ = load_ogb('ogbn-arxiv')
    elif args.dataset == 'ogbn-paper100M':
        g, _ = load_ogb('ogbn-papers100M')
    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(th.sum(g.ndata['train_mask']),
                                                  th.sum(g.ndata['val_mask']),
                                                  th.sum(g.ndata['test_mask'])))
    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    partition_graph(args, g)
