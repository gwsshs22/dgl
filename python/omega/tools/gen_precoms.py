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

    assert training_config["graph_name"] == part_config["graph_name"]
    assert part_config["num_parts"] == 1
    model = model.to(device)
    num_layers = training_config["num_layers"]
    num_hiddens = training_config["num_hiddens"]

    if dataset_config.multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    g = dgl.load_graphs(str(part_config_dir / "part0" / "graph.dgl"))[0][0]
    g = g.to(device)

    node_feats = dgl.data.load_tensors(str(part_config_dir / "part0" / "node_feat.dgl"))
    h = node_feats["_N/features"].to(device)
    labels = node_feats["_N/labels"].to(device)
    pes = []

    for layer_idx in range(num_layers):
        h = model.layer_foward(layer_idx, g, h)
        h.retain_grad()

        if layer_idx != num_layers - 1:
            pes.append(h)

    loss = loss_fn(h, labels)
    loss.backward()

    remaining_test_mask = torch.logical_and(
        node_feats["_N/test_mask"],
        torch.logical_not(node_feats["_N/infer_target_mask"])
    )

    # Sanity check
    print(cal_metrics(
        labels.cpu().numpy()[remaining_test_mask],
        h.detach().cpu().numpy()[remaining_test_mask],
        dataset_config.multilabel))

    pe_grads = [p.grad.cpu() for p in pes]
    pes = [p.detach().cpu() for p in pes]

    data_dir = part_config_dir / "part0" / "data" / training_id
    data_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "pes": pes,
        "pe_grads": pe_grads
    }, data_dir / "tensors.pth")

    # tensor_dict = {}
    # for layer_idx, pe in enumerate(pes):
    #     tensor_dict[f"pe_{layer_idx}"] = pe

    # for layer_idx, pe_grad in enumerate(pe_grads):
    #     tensor_dict[f"pe_grad_{layer_idx}"] = pe_grad

    # data_dir = part_config_dir / "part0" / "data" / training_id
    # data_dir.mkdir(parents=True, exist_ok=True)
    # dgl.data.save_tensors(
    #     str(data_dir / "tensors.dgl"),
    #     tensor_dict
    # )

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
