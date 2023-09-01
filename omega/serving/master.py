import argparse
import queue
import sys
import time
from pathlib import Path
import os
import json

import torch
import torch.distributed.rpc as rpc
import numpy as np

import dgl

from dgl.omega.trace import trace_me, get_traces, put_trace, enable_tracing

from infer_requests import create_req_generator
from worker import WorkerAsyncExecContext, ModelConfig
from worker_comm import create_worker_communicators

class RequestDoneContext:

    def __init__(self):
        self._req_counts = 0
        self._done_counts = 0
        self._error_makred = False
        self._ex = None
        self._exp_started = None
        self._exp_finished = None

    def inc_req(self):
        if self._req_counts == 0:
            self._exp_started = time.time()

        self._req_counts += 1

    def inc_done(self):
        self._done_counts += 1
        if self._done_counts == self._req_counts:
            self._exp_finished = time.time()

    def finished(self):
        return self._req_counts == self._done_counts or self._error_makred

    def mark_error(self, ex):
        self._ex = ex
        self._error_makred = True

    def error_marked(self):
        return self._error_makred

    def get_ex(self):
        return self._ex

    def get_elapsed_time(self):
        return self._exp_finished - self._exp_started

def main(args):
    num_omega_groups = args.num_omega_groups
    num_machines = args.num_machines
    num_gpus_per_machine = args.num_gpus_per_machine
    world_size = num_machines * num_gpus_per_machine
    part_config_path = args.part_config_path
    exec_mode = args.exec_mode

    os.environ["MASTER_ADDR"] = str(args.master_ip)
    os.environ["MASTER_PORT"] = str(args.master_rpc_port)

    rpc.init_rpc("master", rank=0, world_size=world_size * num_omega_groups + 1 + num_machines)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    req_generator = create_req_generator(
        args.trace_dir,
        -1.0 if args.exp_type == "latency" else args.req_per_sec,
        args.arrival_type,
        args.random_seed,
        args.feature_dim)

    model_config = ModelConfig(
        gnn=args.gnn,
        num_inputs=args.num_inputs,
        num_hiddens=args.num_hiddens,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        fanouts=args.fanouts,
    )

    master_dist_comm_ports = [int(p) for p in args.master_dist_comm_ports.split(",")]
    assert len(master_dist_comm_ports) == args.num_omega_groups
    worker_async_exec_context_groups = [
        [
            rpc.remote(
                f"worker-{world_size * omega_group_id + worker_idx}",
                WorkerAsyncExecContext,
                args=(
                    args.ip_config,
                    args.net_type,
                    args.master_ip,
                    master_dist_comm_ports[omega_group_id],
                    num_machines,
                    num_gpus_per_machine,
                    args.worker_num_sampler_threads,
                    args.part_config_path,
                    omega_group_id,
                    Path(args.part_config_path).stem,
                    worker_idx // num_gpus_per_machine,
                    worker_idx,
                    worker_idx % num_gpus_per_machine,
                    exec_mode,
                    args.use_precoms,
                    model_config,
                    args.random_seed,
                    args.profiling,
                    args.tracing
                )
            )
            for worker_idx in range(world_size)
        ] for omega_group_id in range(args.num_omega_groups)
    ]

    worker_comms = create_worker_communicators(
        num_machines,
        num_gpus_per_machine,
        part_config_path,
        exec_mode,
        worker_async_exec_context_groups)

    if args.tracing:
        enable_tracing()

    if args.exp_type == "throughput":
        num_warmups = world_size * num_omega_groups * 2
        latencies = run_throughput_exp(worker_comms, num_warmups, req_generator)
    elif args.exp_type == "latency":
        num_warmups = 4
        latencies = run_latency_exp(worker_comms, num_warmups, req_generator)
    else:
        raise f"Unknown exp_type={exp_type}"
    print("Master finished.", flush=True)

    traces = []
    if args.tracing:
        traces = get_traces("master")

        for worker_async_exec_contexts in worker_async_exec_context_groups:
            for async_exec_context in worker_async_exec_contexts:
                traces += async_exec_context.rpc_sync().collect_traces()

    for worker_async_exec_contexts in worker_async_exec_context_groups:
        for async_exec_context in worker_async_exec_contexts:
            async_exec_context.rpc_sync().shutdown()

    rpc.shutdown()
    print("Master shutdowned.", flush=True)

    if args.result_dir:
        write_result(args, num_warmups, latencies, traces)

def run_throughput_exp(worker_comms, num_warmups, req_generator):
    # Warm-ups
    warm_up_futs = []
    num_worker_comms = len(worker_comms)
    req_generator.set_num_reqs(num_warmups)
    batch_id = 0
    for batch_req in req_generator:
        comm_id = batch_id % num_worker_comms
        fut = worker_comms[comm_id].request(batch_id, batch_req)
        warm_up_futs.append(fut)
        batch_id += 1

    assert num_warmups == len(warm_up_futs)
    print(f"Waiting for {num_warmups} warmup requests.", flush=True)
    torch.futures.wait_all(warm_up_futs)
    print("Warmup done.", flush=True)
    time.sleep(2)

    done_context = RequestDoneContext()
    latencies = []

    req_start_t = time.time()

    num_reqs = int(args.req_per_sec * args.exp_secs)

    req_generator.set_num_reqs(num_reqs)
    req_counts = 0
    for batch_req in req_generator:
        comm_id = batch_id % num_worker_comms
        done_context.inc_req()
        fut = worker_comms[comm_id].request(batch_id, batch_req)
        def done_callback(fut, start_t=time.time()):
            try:
                ret = fut.wait()
                if isinstance(ret, list):
                    batch_id, result_tensor = ret[0].value()
                else:
                    batch_id, result_tensor = ret
                done_context.inc_done()
                done_time = time.time()

                latencies.append(done_time - start_t)

            except Exception as ex:
                done_context.mark_error(ex)
        fut.add_done_callback(done_callback)
        req_counts += 1
        batch_id += 1

    while not done_context.finished():
        time.sleep(0.5)

    latencies = np.array(latencies)

    print(f"Exp elapsed time = {done_context.get_elapsed_time()}", flush=True)
    latencies = np.array(latencies)
    print(f"mean latency={np.mean(latencies)}")
    print(f"p50 latency={np.percentile(latencies, 50)}")
    print(f"p90 latency={np.percentile(latencies, 90)}")
    print(f"p99 latency={np.percentile(latencies, 99)}")
    if done_context.error_marked():
        print(f"An exception occurred while executing inference requests: {done_context.get_ex()}")

    return latencies

def run_latency_exp(worker_comms, num_warmups, req_generator):
    # Warm-ups
    warm_up_futs = []

    req_generator.set_num_reqs(num_warmups)
    batch_id = 0
    for batch_req in req_generator:
        fut = worker_comms[-1].request(batch_id, batch_req)
        warm_up_futs.append(fut)
        batch_id += 1

    print("Waiting for warmup requests.", flush=True)
    torch.futures.wait_all(warm_up_futs)
    print("Warmup done.", flush=True)
    time.sleep(2)

    latencies = []
    req_generator.set_num_reqs(args.num_reqs)
    req_counts = 0
    for batch_req in req_generator:
        start_t = time.time()
        fut = worker_comms[-1].request(batch_id, batch_req)
        ret = fut.wait()
        done_t = time.time()
        if isinstance(ret, list):
            batch_id, result_tensor = ret[0].value()
        else:
            batch_id, result_tensor = ret
        latency = done_t - start_t
        print(f"batch_id={batch_id} done. Took {latency}s", file=sys.stderr)
        put_trace(batch_id, "latency", latency)
        latencies.append(latency)

        req_counts += 1
        batch_id += 1

    latencies = np.array(latencies)
    print(f"Mean latency = {np.mean(latencies)}s")

    return latencies

def write_result(args, num_warmups, latencies, traces):
    result_dir_path = Path(args.result_dir)
    os.makedirs(str(result_dir_path), exist_ok=True)

    args_dict = vars(args)
    args_dict["num_warmups"] = num_warmups

    with open(result_dir_path / "traces.txt", "w") as f:
        for trace in traces:
            f.write(f"{trace.owner},{trace.batch_id},{trace.name},{trace.elapsed_micro}\n")
    
    with open(result_dir_path / "config.json", "w") as f:
        f.write(json.dumps(args_dict, indent=4, sort_keys=True))
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--net_type', type=str, default='socket',
                        help="backend net type, 'socket' or 'tensorpipe'")
    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--master_rpc_port', type=int)
    parser.add_argument('--master_dist_comm_ports', type=str)
    parser.add_argument("--num_omega_groups", type=int, default=1)
    parser.add_argument('--num_machines', type=int)
    parser.add_argument('--num_gpus_per_machine', type=int)
    parser.add_argument('--part_config_path', type=str, required=True)
    parser.add_argument("--worker_num_sampler_threads", type=int, default=16)
    parser.add_argument('--use_precoms', action="store_true")
    parser.add_argument('--exec_mode', type=str, choices=["dp", "cgp", "cgp-multi"])
    parser.add_argument('--trace_dir', type=str, required=True)

    parser.add_argument('--profiling', action="store_true")
    parser.add_argument('--tracing', action="store_true")
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--feature_dim', type=int)

    parser.add_argument('--exp_type', type=str, choices=["latency", "throughput"], required=True)
    # For latency exp
    parser.add_argument('--num_reqs', type=int)
    # For throughput exp
    parser.add_argument('--req_per_sec', type=float)
    parser.add_argument('--exp_secs', type=float)
    parser.add_argument('--arrival_type', choices=['poisson', 'uniform'], default='poisson')

    # Model configuration
    parser.add_argument('--gnn', type=str, required=True)
    parser.add_argument('--num_inputs', type=int, required=True)
    parser.add_argument('--num_hiddens', type=int, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--gat_heads', type=str, required=True)
    parser.add_argument('--fanouts', type=str, required=True)

    parser.add_argument('--random_seed', type=int, default=5123412)

    args = parser.parse_args()
    print(args)
    main(args)
