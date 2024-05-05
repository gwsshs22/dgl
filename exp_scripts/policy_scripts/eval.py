import argparse
import glob
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import subprocess

from omega.models import load_training_config

def run_shell(command):
    try:
        # Using the shell=True option to specify the command as a plain text string
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)

        # Print the output
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        # Print error message and exit the program if the command fails
        print("Command failed with exit code", e.returncode)
        print("Error output:", e.stderr)
        sys.exit(e.returncode)

def run_eval(training_dir, batch_size, dgl_datasets_root, trace_root, output_root, local_rank):
    training_config = load_training_config(training_dir / "config.json")
    gnn = training_config["gnn"]
    graph_name = training_config["graph_name"]

    if graph_name != "ogbn-papers100M":
        gen_precom_cmd = f"""
    python -m omega.tools.gen_precoms \
    --graph_name {graph_name} \
    --part_config {dgl_datasets_root}/{graph_name}/{graph_name}.json \
    --training_dir {training_dir} \
    --local_rank {local_rank}
    """.strip()
        run_shell(gen_precom_cmd)

    output_dir = output_root / f"{graph_name}_{gnn}_bs{batch_size}"

    analyze_cmd = f"""
python -m omega.tools.recompute_policy_analysis \
  --graph_name {graph_name} \
  --part_config {dgl_datasets_root}/{graph_name}/{graph_name}.json \
  --training_dir {training_dir} \
  --trace_dir {trace_root}/{graph_name}-random-{batch_size} \
  --local_rank {local_rank} \
  --output_dir {output_dir} \
  {'' if not args.thresholds else f'--thresholds {args.thresholds}' }
""".strip()
    run_shell(analyze_cmd)

def main(args):
    eval_models_json = json.loads(Path(args.eval_models).read_text())
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    dgl_datasets_root = Path(args.dgl_datasets_root)
    trace_root = Path(args.trace_root)
    output_root = Path(args.output_root)
    training_root = Path(eval_models_json["training_root"])
    model_rel_paths = eval_models_json["rel_paths"]

    for rel_path in model_rel_paths:
        for batch_size in batch_sizes:
            training_dir = training_root / Path(rel_path)
            run_eval(training_dir, batch_size, dgl_datasets_root, trace_root, output_root, args.local_rank)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_models', required=True)
    parser.add_argument('--dgl_datasets_root', required=True)
    parser.add_argument('--trace_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--batch_sizes', default="1024")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--thresholds", type=str)

    args = parser.parse_args()
    main(args)
