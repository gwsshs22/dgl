import argparse
import os
import json
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import dgl
from dgl import function as fn

from dgl.omega.omega_apis import to_block, sample_edges
from omega.utils import get_dataset_config, load_graph
from omega.utils import load_traces
from omega.utils import cal_metrics
from omega.models import load_model_from

def main(args):
    device = f"cuda:{args.local_rank}"
    model, training_config, dataset_config = load_model_from(args.training_dir)
    model = model.to(device)

    graph_name = dataset_config.name
    g = load_graph(
        graph_name,
        ogbn_data_root=args.ogbn_data_root,
        saint_data_root=args.saint_data_root
    )
    g = g.to(device)

    num_layers = training_config["num_layers"]

    with torch.no_grad():
        logits = model([g] * num_layers, g.ndata["features"])
        labels = g.ndata["labels"]

        logits = logits[g.ndata["test_mask"]].to("cpu")
        labels = labels[g.ndata["test_mask"]].to("cpu")

        full_test_f1mic, _ = cal_metrics(labels, logits, dataset_config.multilabel)
    
    print(f"Full test f1mic={full_test_f1mic}")

    if "fanouts" in training_config:
        fanouts = training_config["fanouts"]
        fanouts = [int(f) for f in fanouts.split(",")]
    else:
        fanouts = None
    if fanouts is None:
        return

    test_loader = dgl.dataloading.DataLoader(
        g,
        g.ndata["test_mask"].nonzero().reshape(-1),
        dgl.dataloading.MultiLayerNeighborSampler(fanouts),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    logits_arr = []
    labels_arr = []
    with torch.no_grad():
        for input_nids, seeds, blocks in tqdm(test_loader):
            input_features = g.ndata["features"][input_nids]

            h = model.feature_preprocess(input_features)
            h0 = h
            for layer_idx, block in enumerate(blocks):
                h = model.layer_foward(layer_idx, block, h, h0)
            logits = h
            labels = g.ndata["labels"][seeds]

            logits_arr.append(logits.to("cpu"))
            labels_arr.append(labels.to("cpu"))


    sampled_test_f1mic, _ = cal_metrics(torch.concat(labels_arr), torch.concat(logits_arr), dataset_config.multilabel)
    print(f"Sampled test f1mic={sampled_test_f1mic}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir", required=True)
    parser.add_argument('--ogbn_data_root', required=True)
    parser.add_argument('--saint_data_root', required=True)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--local_rank", type=int, default=0)


    args = parser.parse_args()
    main(args)
