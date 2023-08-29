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
from worker_comm import create_worker_group_communicators

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
    num_machines = args.num_machines
    num_gpus_per_machine = args.num_gpus_per_machine
    world_size = num_machines * num_gpus_per_machine
    part_config_path = args.part_config_path
    exec_mode = args.exec_mode

    os.environ["MASTER_ADDR"] = str(args.master_ip)
    os.environ["MASTER_PORT"] = str(args.master_rpc_port)

    rpc.init_rpc("master", rank=0, world_size=world_size + 1 + num_machines)

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
                args.ip_config,
                args.net_type,
                args.master_ip,
                args.master_dist_comm_port,
                num_machines,
                num_gpus_per_machine,
                args.worker_num_sampler_threads,
                args.part_config_path,
                Path(args.part_config_path).stem,
                i // num_gpus_per_machine,
                i,
                i % num_gpus_per_machine,
                exec_mode,
                args.use_precoms,
                model_config,
                args.random_seed,
                args.tracing))
        for i in range(num_workers)
    ]

    worker_group_comms = create_worker_group_communicators(
        num_machines,
        num_gpus_per_machine,
        part_config_path,
        exec_mode,
        worker_async_exec_contexts)
    num_worker_groups = len(worker_group_comms)

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
    print("Warmup done.", flush=True)

    if args.exp_type == "throughput":
        run_throughput_exp(worker_group_comms, req_generator, batch_id)
    elif args.exp_type == "latency":
        run_latency_exp(worker_group_comms, req_generator, batch_id)
    else:
        raise f"Unknown exp_type={exp_type}"
    print("Master finished.", flush=True)

    for async_exec_context in worker_async_exec_contexts:
        async_exec_context.rpc_sync().shutdown()

    rpc.shutdown()
    print("Master shutdowned.", flush=True)

def run_throughput_exp(worker_group_comms, req_generator, current_batch_id):
    done_context = RequestDoneContext()
    latencies = []

    req_start_t = time.time()

    num_worker_groups = len(worker_group_comms)
    num_reqs = int(args.req_per_sec * args.exp_secs)
    batch_id = current_batch_id
    req_counts = 0
    for batch_req in req_generator:
        group_id = batch_id % num_worker_groups
        done_context.inc_req()
        fut = worker_group_comms[group_id].request(batch_id, batch_req)
        def done_callback(fut, start_t=time.time()):
            try:
                ret = fut.wait()
                if isinstance(ret, list):
                    batch_id = ret[0].value()[0]
                else:
                    batch_id, result_tensor = ret
                done_context.inc_done()
                done_time = time.time()
                tmp = done_time - start_t
                latencies.append(tmp)
                print(f"batch_id={batch_id} latency={tmp} from_start={done_time - req_start_t} done.")

            except Exception as ex:
                done_context.mark_error(ex)
        fut.add_done_callback(done_callback)
        req_counts += 1
        batch_id += 1

        if req_counts == num_reqs:
            break

    while not done_context.finished():
        time.sleep(0.5)

    print(f"Exp elapsed time = {done_context.get_elapsed_time()}", flush=True)
    latencies = np.array(latencies)
    print(f"mean latency={np.mean(latencies)}")
    print(f"p50 latency={np.percentile(latencies, 50)}")
    print(f"p90 latency={np.percentile(latencies, 90)}")
    print(f"p99 latency={np.percentile(latencies, 99)}")
    if done_context.error_marked():
        print(f"An exception occurred while executing inference requests: {done_context.get_ex()}")


def run_latency_exp(worker_group_comms, req_generator, current_batch_id):
    batch_id = current_batch_id
    req_counts = 0
    for batch_req in req_generator:
        start_t = time.time()
        fut = worker_group_comms[-1].request(batch_id, batch_req)
        ret = fut.wait()
        done_t = time.time()
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

    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--net_type', type=str, default='socket',
                        help="backend net type, 'socket' or 'tensorpipe'")
    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--master_rpc_port', type=int)
    parser.add_argument('--master_dist_comm_port', type=int)
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
