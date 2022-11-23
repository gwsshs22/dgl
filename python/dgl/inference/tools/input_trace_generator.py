import argparse
import os
import json
from pathlib import Path
import random

import numpy as np
import torch

import dgl

def find_index_of_values(arr, values):
    sorter = np.argsort(arr)
    return sorter[np.searchsorted(arr, values, sorter=sorter)]

def main(args):
    trace_output_dir = Path(args.output)
    os.makedirs(trace_output_dir, exist_ok=True)

    config_path = Path(os.path.dirname(args.part_config))
    infer_g = dgl.load_graphs(str(config_path / "infer_target_graph.dgl"))[0][0]
    id_mappings = dgl.data.load_tensors(str(config_path / "id_mappings.dgl"))

    infer_target_orig_ids = id_mappings["infer_target_orig_ids"]
    infer_target_features = id_mappings["infer_target_features"]
    orig_ids_in_partitions = id_mappings["orig_ids_in_partitions"]

    new_nodes_start_id = orig_ids_in_partitions.shape[0]

    non_infer_target_ids_in_infer_g = torch.masked_select(torch.arange(infer_g.number_of_nodes()), torch.logical_not(infer_g.ndata["infer_target_mask"]))
    non_infer_target_ids_in_infer_g = set(non_infer_target_ids_in_infer_g.tolist())

    with open(trace_output_dir / "num_traces.txt", "w") as f:
        f.write(f"{args.num_traces}\n")

    for trace_idx in range(args.num_traces):
        selected_index_list = np.random.choice(infer_target_orig_ids.shape[0], (args.batch_size,), replace=False)
        new_orig_ids = infer_target_orig_ids[selected_index_list]

        selected_ids_in_infer_g = find_index_of_values(infer_g.ndata[dgl.NID], new_orig_ids)

        in_edge_eids = infer_g.in_edges(selected_ids_in_infer_g, 'eid')
        out_edge_eids = infer_g.out_edges(selected_ids_in_infer_g, 'eid')

        target_eids = np.union1d(in_edge_eids, out_edge_eids)
        u, v = infer_g.find_edges(target_eids)

        assert(infer_g.ndata["infer_target_mask"][selected_ids_in_infer_g].all())
        filter_set = non_infer_target_ids_in_infer_g.union(set(selected_ids_in_infer_g.tolist()))

        def get_mask(infer_g_node_ids):
            return torch.tensor(list(map(lambda id: id in filter_set, infer_g_node_ids.tolist())), dtype=torch.bool)      
        
        mask = torch.logical_and(get_mask(u), get_mask(v))
        u = torch.masked_select(u, mask)
        v = torch.masked_select(v, mask)

        src_orig_ids = infer_g.ndata[dgl.NID][u]
        dst_orig_ids = infer_g.ndata[dgl.NID][v]

        new_gnids = find_index_of_values(orig_ids_in_partitions, new_orig_ids)
        new_features = infer_target_features[selected_index_list]
        src_gnids = find_index_of_values(orig_ids_in_partitions, src_orig_ids)
        dst_gnids = find_index_of_values(orig_ids_in_partitions, dst_orig_ids)

        new_gnids_map = {}
        for i, new_gnid in enumerate(new_gnids.tolist()):
            new_gnids_map[new_gnid] = i + new_nodes_start_id

        def change_new_gnid(gnids):
            ret = []
            for gnid in gnids.tolist():
                v = new_gnids_map.get(gnid, None)
                if v:
                    ret.append(v)
                else:
                    ret.append(gnid)
            return torch.tensor(ret, dtype=torch.int64)

        dgl.data.save_tensors(str(trace_output_dir / f"{trace_idx}.dgl"), {
            "new_gnids": torch.arange(new_nodes_start_id, new_nodes_start_id + new_gnids.shape[0], dtype=torch.int64),
            "new_features": new_features.type(torch.float32),
            "src_gnids": change_new_gnid(src_gnids),
            "dst_gnids": change_new_gnid(dst_gnids)
        })

        if trace_idx % 100 == 0:
            print(f"{trace_idx}-th input trace generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--part_config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_traces', type=int, required=True)
    parser.add_argument('--random_seed', type=int, default=451241)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    main(args)
