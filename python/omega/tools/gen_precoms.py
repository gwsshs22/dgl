import argparse
import os
import json
from pathlib import Path
import random

import numpy as np
import torch

import dgl

from omega.utils import cal_metrics
from omega.models import load_model_from

def main(args):
    device = f"cuda:{args.local_rank}"
    model, training_config, dataset_config = load_model_from(args.training_dir)
    training_id = training_config["id"]
    part_config_path = Path(args.part_config)
    part_config = json.loads(part_config_path.read_text())
    part_config_dir = part_config_path.parent

    data_dir = part_config_dir / "part0" / "data" / training_id
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "tensors.pth"
    if data_path.exists():
        print(f"{data_path} already exists.")
        return

    assert training_config["graph_name"] == part_config["graph_name"]
    assert part_config["num_parts"] == 1
    model = model.to(device)
    model.eval()
    num_layers = training_config["num_layers"]
    num_hiddens = training_config["num_hiddens"]

    g = dgl.load_graphs(str(part_config_dir / "part0" / "graph.dgl"))[0][0]
    g = g.to(device)

    node_feats = dgl.data.load_tensors(str(part_config_dir / "part0" / "node_feat.dgl"))
    h = node_feats["_N/features"].to(device)
    labels = node_feats["_N/labels"].to(device)
    infer_target_mask = node_feats["_N/infer_target_mask"].type(torch.bool)
    pes = []

    with torch.no_grad():
        h0 = model.feature_preprocess(h)
        h = h0
        for layer_idx in range(num_layers):
            h = model.layer_foward(layer_idx, g, h, h0)

            if layer_idx != num_layers - 1:
                pes.append(h)

    remaining_test_mask = torch.logical_and(
        node_feats["_N/test_mask"].type(torch.bool),
        torch.logical_not(infer_target_mask)
    )

    labels = labels.cpu().clone()
    logits = h.clone().cpu()

    # Sanity check
    print(cal_metrics(
        labels[remaining_test_mask],
        logits[remaining_test_mask],
        dataset_config.multilabel))

    pes = [p.cpu() for p in pes]

    torch.save({
        "pes": pes
    }, data_dir / "tensors.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--part_config', type=str, required=True)
    parser.add_argument('--training_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=451241)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    main(args)
