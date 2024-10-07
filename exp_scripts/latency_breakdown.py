import argparse
from dataclasses import dataclass
import glob
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Any

def is_leaf_dir(target_dir):
    return not any(d.is_dir() for d in target_dir.iterdir())

TRACE_COLUMN_NAMES = [
    "latency",
    "transfer",
    "cg",
    "fetch",
    "copy",
    "gpu_compute",
    "gpu_comm",
    "fetch_size",
    "gpu_all_gather_size",
    "gpu_all_to_all_size"
]

@dataclass
class TraceSummary:
    batch_id: int
    latency_p50: Any
    latency_p90: Any
    latency_p95: Any
    latency_p99: Any
    latency: Any
    transfer: Any
    cg: Any
    fetch: Any
    copy: Any
    gpu_compute: Any
    gpu_comm: Any
    fetch_size: Any
    gpu_all_gather_size: Any
    gpu_all_to_all_size: Any

def _summary_traces(exp_config, trace_path):
    num_warmups = exp_config["num_warmups"]
    traces = []
    for line in trace_path.read_text().split("\n"):
        if not line:
            continue
        tokens = line.split(",")
        traces.append((tokens[0], int(tokens[1]), tokens[2], int(tokens[3])))

    per_batch_data = defaultdict(lambda: defaultdict(dict))
    for owner, batch_id, name, data in traces:
        if batch_id < num_warmups:
            continue
        if name in per_batch_data[batch_id][owner]:
            per_batch_data[batch_id][owner][name].append(data)
        else:
            per_batch_data[batch_id][owner][name] = [data]

    latencies = []
    summaries = []
    for batch_id, batch_data in per_batch_data.items():
        agg_data = defaultdict(list)
        for owner, owner_data in batch_data.items():
            for name, value_list in owner_data.items():
                agg_data[name].append(np.sum(value_list))
        
        agg_data = {
            name: np.mean(data)
            for name, data in agg_data.items()
        }

        latency = agg_data["latency"]
        copy = agg_data["copy"]
        delete = agg_data["delete"]

        fetch = sum([agg_data[k] for k in filter(lambda k: k.startswith("fetch_") and not k.startswith("fetch_size_"), agg_data.keys())])
        fetch_size = sum([agg_data[k] for k in filter(lambda k: k.startswith("fetch_size_"), agg_data.keys())])
        cg = agg_data["mfg"] - fetch
        fetch += delete

        gpu_all_gather = sum([agg_data[k] for k in filter(lambda k: "all_gather" in k and not ("all_gather_size" in k), agg_data.keys())])
        gpu_all_gather_size = sum([agg_data[k] for k in filter(lambda k: "all_gather_size" in k, agg_data.keys())])

        gpu_all_to_all = sum([agg_data[k] for k in filter(lambda k: "all_to_all" in k and not ("all_to_all_size" in k), agg_data.keys())])
        gpu_all_to_all_size = sum([agg_data[k] for k in filter(lambda k: "all_to_all_size" in k, agg_data.keys())])

        gpu_comm = gpu_all_gather + gpu_all_to_all
        gpu_compute = agg_data["compute"] - gpu_comm

        transfer = latency - (copy + cg + fetch + gpu_compute + gpu_comm)

        latencies.append(latency)
        summaries.append(
            TraceSummary(
                batch_id=batch_id,
                latency_p50=None,
                latency_p90=None,
                latency_p95=None,
                latency_p99=None,
                latency=latency,
                transfer=transfer,
                cg=cg,
                fetch=fetch,
                copy=copy,
                gpu_compute=gpu_compute,
                gpu_comm=gpu_comm,
                fetch_size=fetch_size,
                gpu_all_gather_size=gpu_all_gather_size,
                gpu_all_to_all_size=gpu_all_to_all_size
            )
        )

    latency_threshold = np.percentile(latencies, 90)
    summaries = list(filter(lambda s: s.latency <= latency_threshold, summaries))


    tmp_data = defaultdict(list)
    for s in summaries:
        for c in TRACE_COLUMN_NAMES:
            tmp_data[c].append(getattr(s, c))

    def stat(column_name):
        return (np.mean(tmp_data[column_name]), np.std(tmp_data[column_name]))

    return TraceSummary(
        batch_id=0,
        latency_p50=np.percentile(tmp_data["latency"], 50),
        latency_p90=np.percentile(tmp_data["latency"], 90),
        latency_p95=np.percentile(tmp_data["latency"], 95),
        latency_p99=np.percentile(tmp_data["latency"], 99),
        latency=stat("latency"),
        transfer=stat("transfer"),
        cg=stat("cg"),
        fetch=stat("fetch"),
        copy=stat("copy"),
        gpu_compute=stat("gpu_compute"),
        gpu_comm=stat("gpu_comm"),
        fetch_size=stat("fetch_size"),
        gpu_all_gather_size=stat("gpu_all_gather_size"),
        gpu_all_to_all_size=stat("gpu_all_to_all_size")
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
            "graph_name",
            "gnn",
            "num_layers",
            "num_inputs",
            "num_hiddens",
            "num_classes",
            "fanouts",
            "exec_type",
            "recom_threshold",
            "batch_size",
            "feature_cache_size",
            "req_per_sec",
            "throughput",
            "latency_p50",
            "latency_p90",
            "latency_p95",
            "latency_p99",
            "latency",
            "transfer",
            "cg",
            "fetch",
            "copy",
            "gpu_compute",
            "gpu_comm",
            "fetch_size",
            "gpu_all_gather_size",
            "gpu_all_to_all_size",
            "latency_std",
            "transfer_std",
            "cg_std",
            "fetch_std",
            "copy_std",
            "gpu_compute_std",
            "gpu_comm_std",
            "fetch_size_std",
            "gpu_all_gather_size_std",
            "gpu_all_to_all_size_std",
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
                exp_config["graph_name"] if "graph_name" in exp_config else "",
                exp_config["gnn"],
                exp_config["num_layers"],
                exp_config["num_inputs"],
                exp_config["num_hiddens"],
                exp_config["num_classes"],
                exp_config["fanouts"].replace(",", "_"),
                exec_type,
                exp_config["recom_threshold"],
                exp_config["batch_size"],
                exp_config["feature_cache_size"] if "feature_cache_size" in exp_config else "",
                exp_config["req_per_sec"] if "req_per_sec" in exp_config else "",
                exp_config["throughput"] if "throughput" in exp_config else "",
                trace_summary.latency_p50,
                trace_summary.latency_p90,
                trace_summary.latency_p95,
                trace_summary.latency_p99,
                trace_summary.latency[0],
                trace_summary.transfer[0],
                trace_summary.cg[0],
                trace_summary.fetch[0],
                trace_summary.copy[0],
                trace_summary.gpu_compute[0],
                trace_summary.gpu_comm[0],
                trace_summary.fetch_size[0],
                trace_summary.gpu_all_gather_size[0],
                trace_summary.gpu_all_to_all_size[0],
                trace_summary.latency[1],
                trace_summary.transfer[1],
                trace_summary.cg[1],
                trace_summary.fetch[1],
                trace_summary.copy[1],
                trace_summary.gpu_compute[1],
                trace_summary.gpu_comm[1],
                trace_summary.fetch_size[1],
                trace_summary.gpu_all_gather_size[1],
                trace_summary.gpu_all_to_all_size[1],
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
