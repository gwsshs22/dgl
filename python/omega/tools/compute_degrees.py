import argparse
import os
import sys
import json
from pathlib import Path

import dgl
import torch
import numpy as np

from omega.utils import get_dataset_config

def main(args):
    data_dir = Path(args.data_dir)
    dataset_config = get_dataset_config(args.dataset)
    part_config = json.loads((data_dir / f'{args.dataset}.json').read_text())

    num_parts = part_config["num_parts"]
    num_nodes = part_config["num_nodes"]

    in_degrees = torch.zeros(num_nodes, dtype=torch.long)
    dataset_config.undirected = False

    if not dataset_config.undirected:
        out_degrees = torch.zeros(num_nodes, dtype=torch.long)
    else:
        out_degrees = None

    for part_idx in range(num_parts):
        partg = dgl.load_graphs(str(data_dir / f'part{part_idx}' / 'graph.dgl'))[0][0]
        in_degrees[partg.ndata[dgl.NID]] += partg.in_degrees()
        if not dataset_config.undirected:
            out_degrees[partg.ndata[dgl.NID]] += partg.out_degrees()

    degrees = {
        "in_degrees": in_degrees
    }

    if not dataset_config.undirected:
        degrees["out_degrees"] = out_degrees

    dgl.data.save_tensors(str(data_dir / "degrees.dgl"), degrees)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)