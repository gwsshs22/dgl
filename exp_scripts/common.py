import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass

from omega.utils import get_dataset_config


HIDDEN_DIMS = {
    "reddit": 128,
    "yelp": 512,
    "amazon": 512,
    "ogbn-products": 128,
    "ogbn-papers100M": 512,
    "fb5b": 128,
    "fb10b": 128
}

@dataclass
class LatencyExpParams:
    num_reqs: int

@dataclass
class ThroughputExpParams:
    req_per_sec: float
    exp_secs: float
    arrival_type: str = "poisson"

def get_recom_threshold(graph_name, gnn, num_layers):
    if graph_name == "amazon":
        if gnn == "gcn" and num_layers == 2:
            return 98
        if gnn == "gat" and num_layers == 3:
            return 99
    if graph_name == "yelp":
        if gnn == "gcn" and num_layers == 2:
            return 75
        if gnn == "gat" and num_layers == 3:
            return 93
    return 100

def run_exp(
    num_machines,
    graph_name,
    gnn,
    num_layers,
    fanouts,
    exec_type, # dp, dp-precoms, cgp-multi, cgp,
    exp_type, # latency, throughput
    recom_threshold=100, # Use 'auto' for recomputation with < 1% acc drop
    latency_exp_params=None,
    throughput_exp_params=None,
    exp_result_dir=None,
    gat_heads=None,
    feature_dim=None,
    num_hiddens=None,
    batch_size=1024,
    num_gpus_per_machine=2,
    graph_partitioning="random",
    worker_num_sampler_threads=1,
    extra_env_names=[],
    profiling=False,
    force_cuda_mem_uncached=False
):
    extra_envs = " ".join([f"{key}={os.environ[key]}" for key in extra_env_names])
    if force_cuda_mem_uncached:
        extra_envs += " PYTORCH_NO_CUDA_MEMORY_CACHING=1"
    dataset_config = get_dataset_config(graph_name)
    num_classes = dataset_config.num_classes
    num_inputs = dataset_config.num_inputs

    if (graph_name == "fb5b" or graph_name == "fb10b") and feature_dim is None:
        feature_dim = 1024

    if num_hiddens is None:
        num_hiddens = HIDDEN_DIMS[graph_name]
    
    if recom_threshold == 'auto':
        recom_threshold = get_recom_threshold(graph_name, gnn, num_layers)
    
    assert 0 <= recom_threshold and recom_threshold <= 100 and type(recom_threshold) == int

    if feature_dim is not None:
        num_inputs = feature_dim
        num_inputs_args = f" --num_inputs {num_inputs} --feature_dim {num_inputs} "
    else:
        num_inputs_args = f" --num_inputs {num_inputs} "
    
    if fanouts:
        assert len(fanouts) == num_layers
        assert all([f > 0 for f in fanouts])
        fanouts_str = ",".join([str(f) for f in fanouts])
        sampling = True
    else:
        fanouts_str = ",".join(["0"] * num_layers)
        sampling = False

    if gat_heads:
        assert len(gat_heads) == num_layers
        assert all([h > 0 for h in gat_heads])
        gat_heads_str = ",".join([str(h) for h in gat_heads])
    else:
        gat_heads_str = ",".join(["8"] * num_layers)

    partitioned_graph_name = f"{graph_name}-{graph_partitioning}-{num_machines}"
    if exec_type == "cgp" or exec_type == "cgp-multi":
        partitioned_graph_name += "-outedges"

    input_trace_dir = f"$DGL_DATA_HOME/omega_traces/{partitioned_graph_name}-bs-{batch_size}"
    if sampling:
        input_trace_dir += "-sampled"

    if exec_type == "dp":
        exec_args = f" --exec_mode dp "
    elif exec_type == "dp-precoms":
        exec_args = f" --exec_mode dp --use_precoms --recom_threshold {recom_threshold}"
    elif exec_type == "cgp-multi":
        exec_args = f" --exec_mode cgp-multi --use_precoms --recom_threshold {recom_threshold}"
    elif exec_type == "cgp":
        exec_args = f" --exec_mode cgp --use_precoms --recom_threshold {recom_threshold}"
    else:
        raise f"Unkown exec_type {exec_type}"
    
    if exp_type == "latency":
        assert latency_exp_params is not None
        assert latency_exp_params.num_reqs > 0
        exp_args = f" --exp_type latency --num_reqs {latency_exp_params.num_reqs} "
    elif exp_type == "throughput":
        assert throughput_exp_params is not None
        assert throughput_exp_params.req_per_sec > 0.0
        assert throughput_exp_params.exp_secs > 0.0
        assert throughput_exp_params.arrival_type in ["poisson", "uniform"]

        exp_args = f" --exp_type throughput --req_per_sec {throughput_exp_params.req_per_sec} --exp_secs {throughput_exp_params.exp_secs} --arrival_type {throughput_exp_params.arrival_type} "
    else:
        raise f"Unkown exp_type {exp_type}"

    if exp_result_dir:
        exp_result_args = f" --tracing --result_dir {exp_result_dir} "
    else:
        exp_result_args = ""
    
    if profiling:
        profiling_args = f" --profiling "
    else:
        profiling_args = ""

    command = f"""
    python $DGL_HOME/python/omega/tools/launch_omega.py \
    --dgl_home $DGL_HOME \
    --python_bin `which python` \
    --workspace $DGL_DATA_HOME \
    --num_gpus_per_machine {num_gpus_per_machine} \
    --part_config omega_datasets-{num_machines}/{partitioned_graph_name}/{graph_name}.json \
    --extra_envs \
        TP_SOCKET_IFNAME=$DGL_IFNAME \
        GLOO_SOCKET_IFNAME=$DGL_IFNAME \
        NCCL_SOCKET_IFNAME=$DGL_IFNAME \
        DGL_HOME=$DGL_HOME \
        DGL_LIBRARY_PATH=$DGL_HOME/build \
        DGLBACKEND=$DGLBACKEND \
        PYTHONPATH=$DGL_HOME/python {extra_envs} \
    --ip_config ip_configs/ip_config-{num_machines}.txt \
    --worker_num_sampler_threads {worker_num_sampler_threads} \
    --trace_dir {input_trace_dir} \
    --gnn {gnn} --num_layers {num_layers} {num_inputs_args} --num_hiddens {num_hiddens} --num_classes {num_classes} \
    --gat_heads {gat_heads_str} --fanouts {fanouts_str} \
    {exec_args} \
    {exp_args} \
    {exp_result_args} \
    {profiling_args}
    """

    OMEGA_DEBUG = os.environ.get("OMEGA_DEBUG", "0")
    if OMEGA_DEBUG == "1":
        print(f"[DEBUG] command={command}", file=sys.stderr)
    elif exp_has_been_done(exp_result_dir):
        print(f"SKip running as it has been done. command={command}", file=sys.stderr)
    else:
        exit_code = os.system(command)
        if exit_code != 0:
            print(f"Run experiment failed. command={command}", file=sys.stderr)
        if exec_type == "dp":
            time.sleep(30) # Wait for socket release
        else:
            time.sleep(15) # Wait for socket release

def exp_has_been_done(exp_result_dir):
    if not exp_result_dir:
        return False
    exp_result_dir = Path(exp_result_dir)
    return (exp_result_dir / "config.json").exists() and (exp_result_dir / "traces.txt").exists()

if __name__ == "__main__":
    run_exp(
        4,
        "ogbn-papers100M",
        "gat",
        3,
        [],
        "cgp-multi",
        "latency", latency_exp_params=LatencyExpParams(num_reqs=30),
        extra_env_names=[],
        recom_threshold=100,
        )
