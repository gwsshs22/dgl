import argparse
from dataclasses import dataclass
import heapq
import os
import sys
import threading
import queue
import json
import time
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.distributed.rpc as rpc
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

import dgl
from dgl.distributed.kvstore import get_kvstore
from dgl.omega.dist_context import set_nccl_group
from dgl.omega.omega_apis import (
    get_num_assigned_targets_per_gpu,
    to_block,
    to_distributed_block,
)
from dgl.omega.sampler import create_sampler_pool

from dgl.omega.trace import trace_me, get_traces, get_cpp_traces, enable_tracing, put_trace

import dgl.backend as F

from omega.serving.graph_server import OmegaGraphServer
from omega.models import create_model

@dataclass(frozen=True)
class ModelConfig:
    gnn: str
    num_inputs: int
    num_hiddens: int
    num_classes: int
    num_layers: int
    gat_heads: str
    fanouts: str

class WorkerAsyncExecContext:

    def __init__(
        self,
        ip_config,
        net_type,
        master_ip,
        master_dist_comm_port,
        num_machines,
        num_gpus_per_machine,
        worker_num_sampler_threads,
        part_config,
        omega_group_id,
        graph_name,
        machine_rank,
        global_rank,
        local_rank,
        exec_mode,
        use_precoms,
        model_config,
        random_seed,
        profiling,
        tracing):

        self._global_rank = global_rank

        if tracing:
            enable_tracing()

        num_gpus_per_machine_in_group, gpu_ranks, local_gpu_rank_in_group, nid_partitions = self._get_cgp_conf(
            num_machines, machine_rank, num_gpus_per_machine, global_rank, local_rank, exec_mode, part_config)

        fanouts = [int(fanout) for fanout in model_config.fanouts.split(",")]
        fanouts = [-1 if f == 0 else f for f in fanouts]

        self._data_provider = DataProvider(
            ip_config,
            net_type,
            graph_name,
            part_config,
            omega_group_id,
            num_machines,
            machine_rank,
            exec_mode,
            use_precoms,
            model_config.num_layers)

        self._sampler_pool = create_sampler_pool(
            worker_num_sampler_threads,
            num_machines,
            machine_rank,
            num_gpus_per_machine_in_group,
            gpu_ranks,
            local_gpu_rank_in_group,
            nid_partitions,
            exec_mode,
            use_precoms,
            model_config.num_layers,
            fanouts,
            self._data_provider.local_g,
            self._data_provider.local_data_store,
            self._data_provider.pull_fn,
            self._data_provider.dist_sampling_fn)

        self._req_queue = queue.SimpleQueue()
        self._process_thread = threading.Thread(
            target=process_main,
            args=(
                self._req_queue,
                master_ip,
                master_dist_comm_port,
                num_machines,
                num_gpus_per_machine,
                part_config,
                graph_name,
                machine_rank,
                global_rank,
                local_rank,
                exec_mode,
                use_precoms,
                model_config,
                random_seed,
                profiling,
                tracing))
        self._process_thread.start()

    def _get_cgp_conf(self, num_machines, machine_rank, num_gpus_per_machine, global_rank, local_rank, exec_mode, part_config):
        world_size = num_machines * num_gpus_per_machine

        if exec_mode == "cgp-multi":
            num_gpus_per_machine_in_group = 1
            gpu_ranks = [r for r in range(local_rank, world_size + local_rank, num_gpus_per_machine)]
            local_gpu_rank_in_group = 0
        else:
            num_gpus_per_machine_in_group = num_gpus_per_machine
            gpu_ranks = [r for r in range(world_size)]
            local_gpu_rank_in_group = local_rank

        nid_partitions = []
        part_config = json.loads(Path(part_config).read_text())
        assert len(part_config["node_map"]) == 1, "Support only homogeneous graphs currently."
        assert "_N" in part_config["node_map"], "Support only homogeneous graphs currently."
        nid_splits = part_config["node_map"]["_N"]
        nid_partitions = [s[0] for s in nid_splits]
        nid_partitions.append(nid_splits[-1][1])

        return num_gpus_per_machine_in_group, gpu_ranks, local_gpu_rank_in_group, nid_partitions

    @rpc.functions.async_execution
    def process(self, req_id, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        fut = torch.futures.Future()
        worker_arrival_time = time.time()

        def continuation(blocks, src_inputs_list):
            compute_queue_enqueued_time = time.time()
            timestamps = [worker_arrival_time, compute_queue_enqueued_time]
            self._req_queue.put((req_id, batch_id, target_gnids, target_features, blocks, src_inputs_list, timestamps, fut))

        self._sampler_pool.enqueue(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids,
            continuation)

        return fut

    def collect_traces(self):
        python_traces = get_traces(f"worker_{self._global_rank}")
        cpp_traces = get_cpp_traces(f"worker_{self._global_rank}")

        return python_traces + cpp_traces

    def shutdown(self):
        self._req_queue.put((-1,))
        self._process_thread.join()
        self._sampler_pool.shutdown()


class DataProvider:
    def __init__(
        self,
        ip_config,
        net_type,
        graph_name,
        part_config,
        omega_group_id,
        num_machines,
        machine_rank,
        exec_mode,
        use_precoms,
        num_layers):

        dgl.distributed.dist_context.initialize_omega_worker(
            ip_config,
            net_type,
            omega_group_id)

        load_dgl_graph = exec_mode == "dp" and not use_precoms
        self._dist_g = dgl.distributed.DistGraph(graph_name, part_config=part_config, load_dgl_graph=load_dgl_graph)
        self._dist_g.barrier()

        self._local_g = self._dist_g.local_partition
        self._local_data_store = {}

        self._callback_holder = set()

        kv_store = get_kvstore()
        key_names = ["features"]

        if use_precoms:
            for layer_idx in range(num_layers - 1):
                key_names.append(f"layer_{layer_idx}")

        for key in key_names:
            assert f"node~_N~{key}" in kv_store._data_store
            self._local_data_store[key] = kv_store._data_store[f"node~_N~{key}"]
        
        self._num_machines = num_machines
        self._machine_rank = machine_rank
        
        self._graph_servers = []
        for server_rank in range(num_machines):
            if server_rank == self._machine_rank:
                self._graph_servers.append(None)
                continue

            self._graph_servers.append(
                rpc.remote(
                    f"server-{server_rank}",
                    OmegaGraphServer,
                    args=(
                        machine_rank,
                        part_config,
                        use_precoms,
                        num_layers))
            )

    @property
    def local_g(self):
        return self._local_g

    @property
    def local_data_store(self):
        return self._local_data_store

    def pull_fn(self, name, part_ids, local_nids_list):
        part_ids = [p for p in part_ids]
        local_nids_list = [F.zerocopy_from_dgl_ndarray(l) for l in local_nids_list]

        q = queue.SimpleQueue()
        num_reqs = len(part_ids)
        captured_vars = [q, 0, num_reqs]

        def receive_fn():
            captured_vars[1] += 1
            if captured_vars[1] == captured_vars[2]:
                self._callback_holder.remove(receive_fn)
            return captured_vars[0].get()

        for part_id, local_nids in zip(part_ids, local_nids_list):
            fut = self._graph_servers[part_id].rpc_async().remote_pull(name, local_nids)
            def done_callback(f, p=part_id):
                q.put([p, F.zerocopy_to_dgl_ndarray(f.value())])
            fut.add_done_callback(done_callback)

        self._callback_holder.add(receive_fn)
        return receive_fn

    def dist_sampling_fn(self, part_ids, local_nids_list, fanout):
        part_ids = [p for p in part_ids]
        local_nids_list = [F.zerocopy_from_dgl_ndarray(l) for l in local_nids_list]
        futs = []
        for part_id, local_nids in zip(part_ids, local_nids_list):
            fut = self._graph_servers[part_id].rpc_async().remote_sampling(local_nids, fanout)
            futs.append(fut)

        q = queue.SimpleQueue()

        def receive_fn():
            self._callback_holder.remove(receive_fn)
            return q.get()

        def done_callback(f_list):
            ret = []
            for f in f_list.wait():
                r = f.value()
                ret.append(F.zerocopy_to_dgl_ndarray(r[0]))
                ret.append(F.zerocopy_to_dgl_ndarray(r[1]))
            
            q.put(ret)

        torch.futures.collect_all(futs).add_done_callback(done_callback)
        self._callback_holder.add(receive_fn)
        return receive_fn

def process_main(
    req_queue,
    master_ip,
    master_dist_comm_port,
    num_machines,
    num_gpus_per_machine,
    part_config,
    graph_name,
    machine_rank,
    global_rank,
    local_rank,
    exec_mode,
    use_precoms,
    model_config,
    random_seed,
    profiling,
    tracing):

    gpu_ranks = None
    if exec_mode == "cgp" or exec_mode == "cgp-multi":
        world_size = num_machines * num_gpus_per_machine
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_ip}:{master_dist_comm_port}",
            world_size=world_size,
            rank=global_rank)

        if exec_mode == "cgp":
            gpu_ranks = [r for r in range(world_size)]
            set_nccl_group(torch.distributed.new_group())
        else:
            for i in range(num_gpus_per_machine):
                ranks_in_group = [r for r in range(i, world_size + i, num_gpus_per_machine)]
                new_group = torch.distributed.new_group(ranks=ranks_in_group)
                if i == local_rank:
                    gpu_ranks = ranks_in_group
                    set_nccl_group(new_group)

        torch.distributed.barrier()

    exec_context = LocalExecutionContext(
        num_machines,
        num_gpus_per_machine,
        machine_rank,
        global_rank,
        local_rank,
        gpu_ranks,
        exec_mode,
        use_precoms,
        model_config,
        random_seed,
        tracing)

    def run():
        pending_requests = {}
        req_id_heap = []
        expected_req_id = 0

        while True:
            request = req_queue.get()
            req_id = request[0]
            if req_id < 0:
                break

            request[-2].append(time.time()) # compute_queue_dequeded_time

            assert req_id >= expected_req_id
            if expected_req_id == req_id:
                expected_req_id += 1
                exec_context.process_request(*request)
            else:
                # Handle out of order arrivals.
                pending_requests[req_id] = request
                heapq.heappush(req_id_heap, req_id)

            while True:
                if req_id_heap and req_id_heap[0] == expected_req_id:
                    req_id = heapq.heappop(req_id_heap)
                    request = pending_requests[req_id]
                    del pending_requests[req_id]
                    expected_req_id += 1
                    exec_context.process_request(*request)
                else:
                    break

    if profiling:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
        prof.export_chrome_trace(f"trace_{local_rank}_{exec_mode}.json")
    else:
        run()

class LocalExecutionContext:

    def __init__(
        self,
        num_machines,
        num_gpus_per_machine,
        machine_rank,
        global_rank,
        local_rank,
        gpu_ranks,
        exec_mode,
        use_precoms,
        model_config,
        random_seed,
        tracing):

        self._num_machines = num_machines
        self._num_gpus_per_machine = num_gpus_per_machine
        self._machine_rank = machine_rank
        self._global_rank = global_rank
        self._local_rank = local_rank
        self._gpu_ranks = gpu_ranks
        self._exec_mode = exec_mode
        self._use_precoms = use_precoms
        self._model_config = model_config
        self._tracing = tracing

        self._device = f"cuda:{self._local_rank}"

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self._load_model()


    def _load_model(self):
        gat_heads = [int(h) for h in self._model_config.gat_heads.split(",")]
        self._model = create_model(
            self._model_config.gnn,
            self._model_config.num_inputs,
            self._model_config.num_hiddens,
            self._model_config.num_classes,
            self._model_config.num_layers,
            gat_heads)

        self._model = self._model.to(self._device)

    def process_request(self, req_id, batch_id, target_gnids, target_features, blocks, src_inputs_list, timestamps, fut):
        with torch.no_grad():
            try:
                if self._use_precoms:
                    ret_tensor = self.run_with_precoms(batch_id, target_gnids, target_features, blocks, src_inputs_list)
                else:
                    ret_tensor = self.run(batch_id, target_gnids, target_features, blocks, src_inputs_list)
            except Exception as ex:
                fut.set_exception(ex)
                return

            fut.set_result((batch_id, ret_tensor))
            compute_done_time = time.time()
            timestamps.append(compute_done_time)
            self._record_timestamps(batch_id, timestamps)

    def run(self, batch_id, target_gnids, target_features, blocks, src_inputs_list):
        with trace_me(batch_id, "copy", self._device):
            features = src_inputs_list[0]
            blocks = [b.to(self._device) for b in blocks]
            features = features.to(self._device)
            target_features = target_features.to(self._device)

        with trace_me(batch_id, "compute", self._device):
            features = torch.concat((target_features, features))
            h = self._model(blocks, features)

            return h.cpu()

    def run_with_precoms(self, batch_id, target_gnids, target_features, blocks, src_inputs_list):
        with trace_me(batch_id, "copy", self._device):
            num_layers = self._model_config.num_layers

            blocks = [b.to(self._device) for b in blocks]
            inputs = [i.to(self._device) for i in src_inputs_list]
            target_features = target_features.to(self._device)

        with trace_me(batch_id, "compute", self._device):
            h = torch.concat((target_features, inputs[0]))
            for layer_idx in range(num_layers):
                h = self._model.layer_foward(layer_idx, blocks[layer_idx], h)
                if layer_idx != num_layers - 1:
                    h = torch.concat((h, inputs[layer_idx + 1]))

            return h.cpu()
    
    def _record_timestamps(self, batch_id, timestamps):
        if not self._tracing:
            return
        # timestamps = [worker_arrival_time, compute_queue_enqueued_time, compute_queue_dequeded_time, compute_done_time]
        (worker_arrival_time,
            compute_queue_enqueued_time,
            compute_queue_dequeded_time,
            compute_done_time) = timestamps
        process_time = compute_done_time - worker_arrival_time
        sampling_total_time = compute_queue_enqueued_time - worker_arrival_time
        compute_queue_delay_time = compute_queue_dequeded_time - compute_queue_enqueued_time
        compute_total_time = compute_done_time - compute_queue_dequeded_time

        put_trace(batch_id, "process", process_time)
        put_trace(batch_id, "sampling_total", sampling_total_time)
        put_trace(batch_id, "compute_queue_delay", compute_queue_delay_time)
        put_trace(batch_id, "compute_total", compute_total_time)


def main(args):
    num_omega_groups = args.num_omega_groups
    omega_group_id = args.omega_group_id

    num_machines = args.num_machines
    machine_rank = args.machine_rank
    num_gpus_per_machine = args.num_gpus_per_machine
    local_rank = args.local_rank

    world_size = num_machines * num_gpus_per_machine
    rpc_global_rank = world_size * omega_group_id + num_gpus_per_machine * machine_rank + local_rank

    os.environ["MASTER_ADDR"] = str(args.master_ip)
    os.environ["MASTER_PORT"] = str(args.master_rpc_port)

    rpc.init_rpc(
        f"worker-{rpc_global_rank}",
        rank=rpc_global_rank + 1,
        world_size=world_size * num_omega_groups + 1 + num_machines,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            rpc_timeout=600 # 10 minutes timeout
        )
    )
    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--master_rpc_port', type=int)
    parser.add_argument('--master_dist_comm_port', type=int)
    parser.add_argument("--num_omega_groups", type=int, default=1)
    parser.add_argument("--omega_group_id", type=int)
    parser.add_argument('--num_machines', type=int)
    parser.add_argument('--machine_rank', type=int)
    parser.add_argument('--num_gpus_per_machine', type=int)
    parser.add_argument('--local_rank', type=int)

    args = parser.parse_args()

    main(args)
