import argparse
import os
import json
from pathlib import Path
import random

import numpy as np
import torch

import dgl

def find_index_of_values(arr, sorter, values):
    return sorter[np.searchsorted(arr, values, sorter=sorter)]

def main(args):
    sample = args.sample
    trace_dir = Path(args.path)
    sampled_trace_output_dir = trace_dir / f"sample_{sample}"
    os.makedirs(sampled_trace_output_dir, exist_ok=True)
    num_traces = int((trace_dir / "num_traces.txt").read_text())
    with open(sampled_trace_output_dir / "num_traces.txt", "w") as f:
        f.write(f"{num_traces}\n")

    for trace_idx in range(num_traces):
        trace_data = dgl.data.load_tensors(str(trace_dir / f"{trace_idx}.dgl"))

        src_gnids = trace_data['src_gnids']
        dst_gnids = trace_data['dst_gnids']

        g = dgl.graph((src_gnids, dst_gnids))
        sampled_g = g.sample_neighbors(src_gnids.unique(), sample)
        src_gnids, dst_gnids = sampled_g.edges('uv')
        trace_data["src_gnids"] = src_gnids
        trace_data["dst_gnids"] = dst_gnids
        dgl.data.save_tensors(str(sampled_trace_output_dir / f"{trace_idx}.dgl"), trace_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=451241) 
    parser.add_argument("--path", type=str)
    parser.add_argument("--sample", type=int, required=True)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    main(args)
