import argparse
import queue
import sys
import time
from pathlib import Path
import os

import torch
import torch.distributed.rpc as rpc
import numpy as np

import dgl

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
        args.random_seed)

    model_config = ModelConfig(
        gnn=args.gnn,
        num_inputs=args.num_inputs,
        num_hiddens=args.num_hiddens,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        fanouts=args.fanouts,
    )

    master_dist_nccl_ports = [int(p) for p in args.master_dist_nccl_ports.split(",")]
    master_dist_gloo_ports = [int(p) for p in args.master_dist_gloo_ports.split(",")]

    assert len(master_dist_nccl_ports) == args.num_omega_groups
    assert len(master_dist_gloo_ports) == args.num_omega_groups
    worker_async_exec_context_groups = [
        [
            rpc.remote(
                f"worker-{world_size * omega_group_id + worker_idx}",
                WorkerAsyncExecContext,
                args=(
                    args.ip_config,
                    args.net_type,
                    args.master_ip,
                    master_dist_nccl_ports[omega_group_id],
                    master_dist_gloo_ports[omega_group_id],
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

    if args.exp_type == "throughput":
        num_warmups = world_size * num_omega_groups * 2
        run_throughput_exp(worker_comms, num_warmups, req_generator)
    elif args.exp_type == "latency":
        run_latency_exp(worker_comms, req_generator)
    else:
        raise f"Unknown exp_type={exp_type}"
    print("Master finished.", flush=True)

    for worker_async_exec_contexts in worker_async_exec_context_groups:
        for async_exec_context in worker_async_exec_contexts:
            async_exec_context.rpc_sync().shutdown()

    rpc.shutdown()
    print("Master shutdowned.", flush=True)

def wait_for_warmups(warm_up_futs):
    for ret in torch.futures.wait_all(warm_up_futs):
        if isinstance(ret, list):
            batch_id, result_tensor = ret[0].value()
        else:
            batch_id, result_tensor = ret
        print(f"Warmup for batch_id={batch_id} done.", flush=True)

def run_throughput_exp(worker_comms, num_warmups, req_generator):
    # Warm-ups
    warm_up_futs = []
    num_worker_comms = len(worker_comms)
    batch_id = 0
    req_generator.set_num_reqs(num_warmups)
    for batch_req in req_generator:
        comm_id = batch_id % num_worker_comms
        fut = worker_comms[comm_id].request(batch_id, batch_req)
        warm_up_futs.append(fut)
        batch_id += 1

    assert len(warm_up_futs) == num_warmups

    print(f"Waiting for {num_warmups} warmup requests.", flush=True)
    wait_for_warmups(warm_up_futs)
    print("Warmup done.", flush=True)

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
                    batch_id = ret[0].value()[0]
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

        if req_counts == num_reqs:
            break

    while not done_context.finished():
        time.sleep(0.5)
    
    if done_context.error_marked():
        print(f'Error!!! {done_context.get_ex()}', file=sys.stderr)

    latencies = np.array(latencies)

    print(f"Exp elapsed time = {done_context.get_elapsed_time()}", flush=True)
    latencies = np.array(latencies)
    print(f"mean latency={np.mean(latencies)}")
    print(f"p50 latency={np.percentile(latencies, 50)}")
    print(f"p90 latency={np.percentile(latencies, 90)}")
    print(f"p99 latency={np.percentile(latencies, 99)}")
    if done_context.error_marked():
        print(f"An exception occurred while executing inference requests: {done_context.get_ex()}")


def run_latency_exp(worker_comms, req_generator):
    # Warm-ups
    num_warmups = 4
    warm_up_futs = []

    req_generator.set_num_reqs(num_warmups)
    batch_id = 0
    for batch_req in req_generator:
        fut = worker_comms[-1].request(batch_id, batch_req)
        warm_up_futs.append(fut)
        batch_id += 1

    assert len(warm_up_futs) == num_warmups

    print("Waiting for warmup requests.", flush=True)
    wait_for_warmups(warm_up_futs)
    print("Warmup done.", flush=True)

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
        print(f"batch_id={batch_id} done. Took {done_t - start_t}s.", file=sys.stderr)

        req_counts += 1
        batch_id += 1

    assert req_counts == args.num_reqs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--net_type', type=str, default='socket',
                        help="backend net type, 'socket' or 'tensorpipe'")
    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--master_rpc_port', type=int)
    parser.add_argument('--master_dist_nccl_ports', type=str)
    parser.add_argument('--master_dist_gloo_ports', type=str)
    parser.add_argument("--num_omega_groups", type=int, default=1)
    parser.add_argument('--num_machines', type=int)
    parser.add_argument('--num_gpus_per_machine', type=int)
    parser.add_argument('--part_config_path', type=str, required=True)
    parser.add_argument("--worker_num_sampler_threads", type=int, default=16)
    parser.add_argument('--use_precoms', action="store_true")
    parser.add_argument('--exec_mode', type=str, choices=["dp", "cgp", "cgp-multi"])
    parser.add_argument('--trace_dir', type=str, required=True)

    parser.add_argument('--tracing', action="store_true")

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
