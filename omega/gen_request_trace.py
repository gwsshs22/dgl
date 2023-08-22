import argparse
import os
import json
from pathlib import Path
import random

import numpy as np
import torch

import dgl

from dgl.omega.omega_apis import trace_gen_helper

def find_index_of_values(arr, sorter, values):
    return sorter[np.searchsorted(arr, values, sorter=sorter)]

def main(args):
    trace_output_dir = Path(args.output)
    os.makedirs(trace_output_dir, exist_ok=True)

    config_path = Path(os.path.dirname(args.part_config))
    infer_g = dgl.load_graphs(str(config_path / "infer_target_graph.dgl"))[0][0]
    id_mappings = dgl.data.load_tensors(str(config_path / "id_mappings.dgl"))

    orig_nids = id_mappings["orig_nids"]
    orig_nids_sorter = np.argsort(orig_nids)

    first_new_gnid = id_mappings["infer_target_mask"].shape[0] + 1

    infer_target_mask = infer_g.ndata["infer_target_mask"].bool()
    infer_target_local_ids = torch.masked_select(torch.arange(infer_target_mask.shape[0]), infer_target_mask)

    with open(trace_output_dir / "num_traces.txt", "w") as f:
        f.write(f"{args.num_traces}\n")

    infer_target_mask = infer_target_mask.type(torch.int64)
    for trace_idx in range(args.num_traces):
        batch_local_ids = torch.tensor(np.random.choice(infer_target_local_ids, (args.batch_size,), replace=False))
        target_features = infer_g.ndata["features"][batch_local_ids]

        u, v = infer_g.in_edges(batch_local_ids, 'uv')
        u_orig = infer_g.ndata[dgl.NID][u]
        v_orig = infer_g.ndata[dgl.NID][v]

        u_in_partitions = find_index_of_values(orig_nids, orig_nids_sorter, u_orig)
        v_in_partitions = find_index_of_values(orig_nids, orig_nids_sorter, v_orig)

        target_gnids, src_gnids, dst_gnids = trace_gen_helper(
            first_new_gnid, infer_target_mask, batch_local_ids, u, v, u_in_partitions, v_in_partitions)

        dgl.data.save_tensors(str(trace_output_dir / f"{trace_idx}.dgl"), {
            "target_gnids": target_gnids,
            "src_gnids": src_gnids,
            "dst_gnids": dst_gnids,
            "target_features": target_features.type(torch.float32)
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
