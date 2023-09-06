import argparse
import time
import sys
import os
import gc
import json
import glob
from pathlib import Path

import dgl
import numpy as np
import torch

from dgl.omega.omega_apis import (
    partition_facebook_dataset
)

def main(args):
    start_t = time.time()

    num_parts = args.num_parts
    input_dir = args.input_dir
    infer_prob = args.infer_prob


    input_dir = Path(input_dir)
    partition_facebook_dataset(
        num_parts,
        str(input_dir),
        [str(p) for p in input_dir.glob("part-m-*")],
        infer_prob,
        args.num_threads
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global_data = dgl.data.load_tensors(str(input_dir / "global-data.dgl"))
    num_nodes_per_parts = global_data["num_nodes_per_parts"].tolist()
    num_edges_per_parts = []
    orig_ids = global_data["orig_ids"]

    infer_target_mask = global_data["infer_target_mask"].type(torch.bool)
    dgl.data.save_tensors(str(output_dir / "id_mappings.dgl"), {
        "infer_target_mask": infer_target_mask,
        "orig_nids": orig_ids
    })

    for part_id in range(num_parts):
        graph_data = dgl.data.load_tensors(str(input_dir / f"graph-data-{part_id}.dgl"))
        dgl_nids = graph_data["dgl_nids"]
        part_id_arr = graph_data["part_id"]
        src_ids, dst_ids = graph_data["new_src_ids"], graph_data["new_dst_ids"]

        num_nodes = dgl_nids.shape[0]
        part_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes)
        num_inner_nodes = num_nodes_per_parts[part_id]
        num_edges = part_g.number_of_edges()

        inner_node_arr = torch.zeros(num_nodes, dtype=torch.bool)
        inner_node_arr[:num_inner_nodes] = True
        inner_edge_arr = torch.ones(num_edges, dtype=torch.bool)

        part_g.ndata[dgl.NID] = dgl_nids
        part_g.ndata["part_id"] = part_id_arr
        part_g.ndata["inner_node"] = inner_node_arr

        part_g.edata[dgl.NID] = torch.arange(num_edges, dtype=torch.int64)
        part_g.edata["inner_edge"] = inner_edge_arr

        dgl.save_graphs(str(output_dir / f"part{part_id}" / "graph.dgl"), [part_g])
        dgl.data.save_tensors(str(output_dir / f"part{part_id}" / "node_feat.dgl"), {
            "_N/features": torch.rand(num_inner_nodes, args.feature_dim)
        })
        dgl.data.save_tensors(str(output_dir / f"part{part_id}" / "edge_feat.dgl"), {})
        num_edges_per_parts.append(num_edges)

    infer_graph_data = dgl.data.load_tensors(str(input_dir / f"infer-target-graph-data.dgl"))

    dgl_nids = infer_graph_data["dgl_nids"]
    src_ids, dst_ids = infer_graph_data["new_src_ids"], infer_graph_data["new_dst_ids"]
    infer_g = dgl.graph((src_ids, dst_ids), num_nodes=dgl_nids.shape[0])
    infer_g.ndata[dgl.NID] = dgl_nids
    infer_g.ndata["infer_target_mask"] = infer_graph_data["infer_target_mask"]
    infer_g.ndata["features"] = torch.rand(infer_g.number_of_nodes(), args.feature_dim)
    dgl.save_graphs(str(output_dir / "infer_target_graph.dgl"), [infer_g])

    num_nodes_arr = []
    num_edges_arr = []
    s = 0
    for num_node in num_nodes_per_parts:
        e = s + num_node
        num_nodes_arr.append([s, e])
        s = e

    s = 0
    for num_edge in num_edges_per_parts:
        e = s + num_edge
        num_edges_arr.append([s, e])
        s = e
 
    part_config = {
        "edge_map": {
            "_N:_E:_N": num_edges_arr
        },
        "etypes": {
            "_N:_E:_N": 0
        },
        "graph_name": args.dataset,
        "halo_hops": 1,
        "node_map": {
            "_N": num_nodes_arr
        },
        "ntypes": {
            "_N": 0
        },
        "num_edges": num_edges_arr[-1][-1],
        "num_nodes": num_nodes_arr[-1][-1],
        "num_parts": num_parts,
        "part_method": "random",
    }

    for part_id in range(num_parts):
        part_config[f"part-{part_id}"] = {
            "edge_feats": f"part{part_id}/edge_feat.dgl",
            "node_feats": f"part{part_id}/node_feat.dgl",
            "part_graph": f"part{part_id}/graph.dgl"
        }
    
    with open(output_dir / f"{args.dataset}.json", "w") as f:
        json.dump(part_config, f, sort_keys=True, indent=4)

    for p in input_dir.glob("*.dgl"):
        p.unlink()
    
    print(f"Partition Done. Took {time.time() - start_t}s")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition graphs")
    argparser.add_argument('--dataset', type=str, required=True,
                           help='fb5b, fb10b')
    argparser.add_argument('--input_dir', type=str, required=True)
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--infer_prob', type=float, default=0.01)
    argparser.add_argument('--num_threads', type=int, default=64)
    argparser.add_argument('--feature_dim', type=int, default=16)
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.', required=True)

    args = argparser.parse_args()

    main(args)
