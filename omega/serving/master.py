import argparse
from threading import Thread
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
from worker_comm import create_worker_group_communicators

class RequestDoneContext:

    def __init__(self):
        self._req_counts = 0
        self._done_counts = 0
        self._error_makred = False
        self._ex = None

    def inc_req(self):
        self._req_counts += 1

    def inc_done(self):
        self._done_counts += 1

    def finished(self):
        return self._req_counts == self._done_counts or self._error_makred

    def mark_error(self, ex):
        self._ex = ex
        self._error_makred = True

    def error_marked(self):
        return self._error_makred

    def get_ex(self):
        return self._ex

def main(args):
    num_machines = args.num_machines
    num_gpus_per_machine = args.num_gpus_per_machine
    world_size = num_machines * num_gpus_per_machine + 1
    part_config_path = args.part_config_path
    exec_mode = args.exec_mode

    os.environ["MASTER_ADDR"] = str(args.master_ip)
    os.environ["MASTER_PORT"] = str(args.master_rpc_port)

    rpc.init_rpc("master", rank=0, world_size=world_size)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    req_generator = create_req_generator(
        args.trace_dir,
        -1.0 if args.exp_type == "latency" else args.req_per_sec)

    num_workers = num_machines * num_gpus_per_machine
    model_config = ModelConfig(
        gnn=args.gnn,
        num_inputs=args.num_inputs,
        num_hiddens=args.num_hiddens,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        fanouts=args.fanouts,
    )

    worker_async_exec_contexts = [
        rpc.remote(
            f"worker-{i}",
            WorkerAsyncExecContext,
            args=(
                args.master_ip,
                args.master_dist_comm_port,
                num_machines,
                num_gpus_per_machine,
                args.part_config_path,
                Path(args.part_config_path).stem,
                i // num_gpus_per_machine,
                i,
                i % num_gpus_per_machine,
                exec_mode,
                args.use_precoms,
                model_config,
                args.random_seed))
        for i in range(num_workers)
    ]

    worker_group_comms = create_worker_group_communicators(
        num_machines,
        num_gpus_per_machine,
        part_config_path,
        exec_mode,
        worker_async_exec_contexts)
    num_worker_groups = len(worker_group_comms)
    time.sleep(5)
    # Warm-ups
    num_warmups = num_workers * 2
    warm_up_futs = []

    batch_id = 0
    for batch_req in req_generator:
        group_id = batch_id % num_worker_groups
        fut = worker_group_comms[group_id].request(batch_id, batch_req)
        warm_up_futs.append(fut)
        batch_id += 1

        if len(warm_up_futs) == num_warmups:
            break

    print("Waiting for warmup requests.", flush=True)
    torch.futures.wait_all(warm_up_futs)
    print("Warmup done.")

    if args.exp_type == "throughput":
        run_throughput_exp(worker_group_comms, req_generator, batch_id)
    elif args.exp_type == "latency":
        run_latency_exp(worker_group_comms, req_generator, batch_id)
    else:
        raise f"Unknown exp_type={exp_type}"
    rpc.shutdown()

def run_throughput_exp(worker_group_comms, req_generator, current_batch_id):
    done_context = RequestDoneContext()
    def done_callback(fut):
        try:
            ret = fut.wait()
            if isinstance(ret, list):
                batch_id = ret[0].value()[0]
            else:
                batch_id, result_tensor = ret
            print(f"batch_id={batch_id} done.")
            done_context.inc_done()
        except Exception as ex:
            done_context.mark_error(ex)

    num_worker_groups = len(worker_group_comms)
    num_reqs = int(args.req_per_sec * args.exp_secs)
    batch_id = current_batch_id
    req_counts = 0
    for batch_req in req_generator:
        group_id = batch_id % num_worker_groups
        done_context.inc_req()
        fut = worker_group_comms[group_id].request(batch_id, batch_req)
        fut.add_done_callback(done_callback)
        req_counts += 1
        batch_id += 1

        if req_counts == num_reqs:
            break

    while not done_context.finished():
        time.sleep(0.5)

    if done_context.error_marked():
        print(f"An exception occurred while executing inference requests: {done_context.get_ex()}")


def run_latency_exp(worker_group_comms, req_generator, current_batch_id):
    batch_id = current_batch_id
    req_counts = 0
    for batch_req in req_generator:
        start_t = time.perf_counter()
        fut = worker_group_comms[-1].request(batch_id, batch_req)
        ret = fut.wait()
        done_t = time.perf_counter()
        if isinstance(ret, list):
            batch_id = ret[0].value()[0]
        else:
            batch_id, result_tensor = ret
        print(f"batch_id={batch_id} done. Took {done_t - start_t}s", file=sys.stderr)

        req_counts += 1
        batch_id += 1

        if req_counts == args.num_reqs:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--master_rpc_port', type=int)
    parser.add_argument('--master_dist_comm_port', type=int)
    parser.add_argument('--num_machines', type=int)
    parser.add_argument('--num_gpus_per_machine', type=int)
    parser.add_argument('--part_config_path', type=str, required=True)
    parser.add_argument('--use_precoms', action="store_true")
    parser.add_argument('--exec_mode', type=str, choices=["dp", "cgp", "cgp-multi"])
    parser.add_argument('--trace_dir', type=str, required=True)

    parser.add_argument('--exp_type', type=str, choices=["latency", "throughput"], required=True)
    # For latency exp
    parser.add_argument('--num_reqs', type=int)
    # For throughput exp
    parser.add_argument('--req_per_sec', type=float)
    parser.add_argument('--exp_secs', type=float)

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
