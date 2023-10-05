import argparse
from dataclasses import dataclass
import glob
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from omega.models import load_training_config

def is_leaf_dir(target_dir):
    return not any(d.is_dir() for d in target_dir.iterdir())

@dataclass
class TrainingResult:
    graph_name: str
    gnn: str
    num_layers: str
    num_hiddens: int
    fanouts: str
    f1mic: float
    lr: float
    num_epochs: int
    running_epochs: int
    patience: int
    val_every: int
    gcn_norm: str
    

def analyze_result(root_dir, result_dir, training_config_path):
    training_config = load_training_config(training_config_path)
    
    relative_path = result_dir.relative_to(root_dir)

    training_result = TrainingResult(
        graph_name=training_config["graph_name"],
        gnn=training_config["gnn"],
        num_layers=training_config["num_layers"],
        num_hiddens=training_config["num_hiddens"],
        fanouts=training_config["fanouts"].replace("-1", "0").replace(",", "_"),
        f1mic=training_config["test_f1_mic"],
        lr=training_config["lr"],
        num_epochs=training_config["num_epochs"],
        running_epochs=training_config["running_epochs"],
        patience=training_config["patience"],
        val_every=training_config["val_every"],
        gcn_norm=training_config["gcn_norm"]
    )

    return training_config, relative_path, training_result

def main(args):
    result_root_dir = Path(args.result_root_dir)
    results = []
    for result_dir in result_root_dir.rglob("./**/"):
        if not is_leaf_dir(result_dir):
            continue

        training_config_path = Path(result_dir / "config.json")
        if training_config_path.exists():
            results.append(analyze_result(result_root_dir, result_dir, training_config_path))

    # Write latency breakdown summary.
    with open(result_root_dir / "training_results.csv", "w") as f:
        head = ",".join([
            "graph_name",
            "gnn",
            "num_layers",
            "num_hiddens",
            "fanouts",
            "f1mic",
            "lr",
            "gcn_norm",
            "num_epochs",
            "running_epochs",
            "patience",
            "val_every",
            "rel_path"
        ])

        f.write(head)
        f.write("\n")

        for training_config, relative_path, training_result in results:
            if training_config["gnn"] == "gcn" and training_config["gcn_norm"] == "both":
                continue
            row = ",".join([str(c) for c in [
                training_result.graph_name,
                training_result.gnn,
                training_result.num_layers,
                training_result.num_hiddens,
                training_result.fanouts,
                training_result.f1mic,
                training_result.lr,
                training_result.gcn_norm,
                training_result.num_epochs,
                training_result.running_epochs,
                training_result.patience,
                training_result.val_every,
                relative_path
            ]])

            f.write(row)
            f.write("\n")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root_dir', required=True)

    args = parser.parse_args()
    main(args)
