import random
import argparse

import dgl
import torch

def make_subgraph(node_ids):
    ret = []
    for s in node_ids:
        for e in node_ids:
            ret.append([s, e])
    return ret

def main(args):
    part_method = args.part_method
    num_partitions = args.num_partitions
    graph_name = args.graph_name
    num_inputs = args.num_inputs
    random_seed = args.random_seed
    out_path = args.out_path

    random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    edge_set1 = make_subgraph([0, 2, 3, 5, 7])
    edge_set2 = make_subgraph([1, 4, 6, 8, 9])
    edge_set3 = [[3, 1], [3, 4], [7, 6], [6, 3], [6, 2]]
    edge_set = edge_set1 + edge_set2 + edge_set3
    random.shuffle(edge_set)

    start_nodes = list(map(lambda edge: edge[0], edge_set))
    end_nodes = list(map(lambda edge: edge[1], edge_set))
    g = dgl.graph((start_nodes, end_nodes))
    g.ndata['features'] = torch.rand(g.num_nodes(), num_inputs, dtype=torch.float32)
    dgl.distributed.partition_graph(g, graph_name, 2, out_path,
                                    part_method=part_method,
                                    balance_ntypes=None,
                                    balance_edges=False,
                                    num_trainers_per_machine=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_method', type=str, required=True, choices=["random", "metis"])
    parser.add_argument('--num_partitions', type=int, default=2)
    parser.add_argument('--graph_name', type=str, default="mygraph")
    parser.add_argument('--num_inputs', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=15243241)
    parser.add_argument('--out_path', type=str, required=True)

    args = parser.parse_args()
    print(args)

    main(args)


