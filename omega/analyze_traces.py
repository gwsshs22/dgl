import argparse
from dataclasses import dataclass
import glob
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def is_leaf_dir(target_dir):
    return not any(d.is_dir() for d in target_dir.iterdir())

@dataclass
class TraceSummary:
    latency: int
    transfer: int
    fetch: int
    sampling: int
    copy: int
    compute: int

def _summary_traces(exp_config, trace_path):
    num_warmups = exp_config["num_warmups"]
    traces = []
    for line in trace_path.read_text().split("\n"):
        if not line:
            continue
        tokens = line.split(",")
        traces.append((tokens[0], int(tokens[1]), tokens[2], int(tokens[3])))

    per_batch_data = defaultdict(dict)
    for owner, batch_id, name, elapsed_micro in traces:
        if batch_id < num_warmups:
            continue
        if name in per_batch_data[batch_id]:
            per_batch_data[batch_id][name].append(elapsed_micro)
        else:
            per_batch_data[batch_id][name] = [elapsed_micro]

    latencies = []
    fetches = []
    samplings = []
    computes = []
    copies = []
    transfers = []

    for batch_id, batch_data in per_batch_data.items():
        fetch = 0
        if "fetch" in batch_data:
            fetch = np.sum(batch_data["fetch"])

        compute = np.mean(batch_data["compute"])
        compute_queue_delay = np.mean(batch_data["compute_queue_delay"])
        sampling = np.mean(batch_data["sampling"])
        
        assert len(batch_data["latency"]) == 1
        latency = batch_data["latency"][0]

        sampling -= fetch
        sampling += compute_queue_delay

        copy = np.mean(batch_data["copy"])
        transfer = latency - fetch - sampling - compute - copy

        latencies.append(latency)
        fetches.append(fetch)
        samplings.append(sampling)
        computes.append(compute)
        copies.append(copy)
        transfers.append(transfer)

    return TraceSummary(
        latency=np.mean(latencies),
        transfer=np.mean(transfers),
        fetch=np.mean(fetches),
        sampling=np.mean(samplings),
        copy=np.mean(copies),
        compute=np.mean(computes)
    )

def analyze_exp(root_dir, result_dir):
    exp_config = json.loads(Path(result_dir / "config.json").read_text())
    relative_path = result_dir.relative_to(root_dir)

    trace_path = result_dir / "traces.txt"
    trace_summary = _summary_traces(exp_config, trace_path) if trace_path.exists() else None
    
    return exp_config, relative_path, trace_summary


def main(args):
    exp_root_dir = Path(args.exp_root_dir)
    results = []
    for result_dir in exp_root_dir.rglob("./**/"):
        if not is_leaf_dir(result_dir):
            continue

        results.append(analyze_exp(exp_root_dir, result_dir))

    # Write latency breakdown summary.
    with open(exp_root_dir / "latency_breakdown.csv", "w") as f:
        head = ",".join([
            "num_machines",
            "gnn",
            "num_layers",
            "num_inputs",
            "num_hiddens",
            "num_classes",
            "fanouts",
            "exec_type",
            "batch_size",
            "latency",
            "transfer",
            "sampling",
            "fetch",
            "copy",
            "compute",
            "rel_path"
        ])

        f.write(head)
        f.write("\n")

        for exp_config, relative_path, trace_summary in results:
            exec_type = exp_config["exec_mode"]
            if exec_type == "dp" and exp_config["use_precoms"]:
                exec_type = "dp-precoms"

            row = ",".join([str(c) for c in [
                exp_config["num_machines"],
                exp_config["gnn"],
                exp_config["num_layers"],
                exp_config["num_inputs"],
                exp_config["num_hiddens"],
                exp_config["num_classes"],
                exp_config["fanouts"].replace(",", "_"),
                exec_type,
                exp_config["batch_size"],
                trace_summary.latency,
                trace_summary.transfer,
                trace_summary.sampling,
                trace_summary.fetch,
                trace_summary.copy,
                trace_summary.compute,
                relative_path
            ]])

            f.write(row)
            f.write("\n")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)

    args = parser.parse_args()
    main(args)
