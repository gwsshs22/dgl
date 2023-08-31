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
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import torch.multiprocessing as mp
import numpy as np

import dgl
from dgl.distributed.kvstore import get_kvstore
from dgl.omega.sampler import create_sampler_pool
import dgl.backend as F

from gnn_executor import gnn_executor_main
from graph_server import OmegaGraphServer
from utils import init_torch_distributed

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
        master_dist_nccl_port,
        master_dist_gloo_port,
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
        tracing):

        num_gpus_per_machine_in_group, gpu_ranks, local_gpu_rank_in_group, nid_partitions = self._get_cgp_conf(
            num_machines, machine_rank, num_gpus_per_machine, global_rank, local_rank, exec_mode, part_config)

        fanouts = [int(fanout) for fanout in model_config.fanouts.split(",")]
        fanouts = [-1 if f == 0 else f for f in fanouts]

        full_sampling = fanouts[0] == -1
        if full_sampling:
            for f in fanouts:
                assert f == -1

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

        self._req_queue = queue.Queue()
        self._process_thread = threading.Thread(
            target=process_main,
            args=(
                self._req_queue,
                master_ip,
                master_dist_nccl_port,
                master_dist_gloo_port,
                num_machines,
                num_gpus_per_machine,
                omega_group_id,
                part_config,
                graph_name,
                machine_rank,
                global_rank,
                local_rank,
                num_gpus_per_machine_in_group,
                gpu_ranks,
                local_gpu_rank_in_group,
                exec_mode,
                use_precoms,
                model_config,
                full_sampling,
                random_seed,
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

        def continuation(sampling_rets):
            self._req_queue.put((req_id, batch_id, target_features, sampling_rets, fut))

        self._sampler_pool.enqueue(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids,
            continuation)

        return fut

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

        self._dist_g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
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
    master_dist_nccl_port,
    master_dist_gloo_port,
    num_machines,
    num_gpus_per_machine,
    omega_group_id,
    part_config,
    graph_name,
    machine_rank,
    global_rank,
    local_rank,
    num_gpus_per_machine_in_group,
    gpu_ranks,
    local_gpu_rank_in_group,
    exec_mode,
    use_precoms,
    model_config,
    full_sampling,
    random_seed,
    tracing):

    gloo_group, gloo_group_ranks = init_torch_distributed(
        exec_mode,
        num_machines,
        num_gpus_per_machine,
        master_ip,
        master_dist_gloo_port,
        global_rank,
        local_rank,
        "gloo"
    )

    assert gloo_group_ranks == gpu_ranks

    gnn_executor_manager = GnnExecutorManager(
        master_ip,
        master_dist_nccl_port,
        gloo_group,
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        omega_group_id,
        global_rank,
        local_rank,
        num_gpus_per_machine_in_group,
        gpu_ranks,
        local_gpu_rank_in_group,
        exec_mode,
        use_precoms,
        model_config,
        full_sampling,
        random_seed,
        tracing
    )

    pending_requests = {}
    req_id_heap = []
    expected_req_id = 0

    while True:
        request = req_queue.get()
        req_id = request[0]
        if req_id < 0:
            break

        assert req_id >= expected_req_id
        if expected_req_id == req_id:
            expected_req_id += 1
            gnn_executor_manager.send_req(request)
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
                gnn_executor_manager.send_req(request)
            else:
                break

    gnn_executor_manager.shutdown()

class GnnExecutorManager:

    def __init__(
        self,
        master_ip,
        master_dist_comm_port,
        gloo_group,
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        omega_group_id,
        global_rank,
        local_rank,
        num_gpus_per_machine_in_group,
        gpu_ranks,
        local_gpu_rank_in_group,
        exec_mode,
        use_precoms,
        model_config,
        full_sampling,
        random_seed,
        tracing
    ):
        mp.set_start_method("spawn")
        self._gloo_group = gloo_group
        self._local_rank = local_rank
        self._exec_mode = exec_mode
        self._full_sampling = full_sampling
        self._use_precoms = use_precoms
        self._tracing = tracing
        self._device = f"cuda:{local_rank}"

        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self._req_holder = {}

        self._wait_sem = mp.Semaphore(0)
        self._finish_event = mp.Event()

        self._gnn_executor_process = mp.Process(
            target=gnn_executor_main,
            args=(
                self._in_queue,
                self._out_queue,
                master_ip,
                master_dist_comm_port,
                num_machines,
                machine_rank,
                num_gpus_per_machine,
                omega_group_id,
                global_rank,
                local_rank,
                num_gpus_per_machine_in_group,
                gpu_ranks,
                local_gpu_rank_in_group,
                exec_mode,
                use_precoms,
                model_config,
                full_sampling,
                random_seed,
                self._tracing,
                self._device,
                self._wait_sem,
                self._finish_event
            )
        )
        self._gnn_executor_process.start()

        self._callback_thread = threading.Thread(
            target=GnnExecutorManager.callback_thread_main,
            args=(
                self._req_holder,
                self._out_queue,
                self._device,
                self._wait_sem
            )
        )
        self._callback_thread.start()

    def send_req(self, request):
        if self._exec_mode == "dp":
            self.send_req_dp(*request)
        else:
            self.send_req_cgp(*request)

    def send_req_dp(self, req_id, batch_id, target_features, sampling_rets, fut):
        target_features = target_features.to(self._device)
        block_data_list, block_src_inputs_list = self._copy_block_data(sampling_rets)
        gnn_executor_req = (batch_id, target_features, block_data_list, None, block_src_inputs_list)
        self._req_holder[batch_id] = (gnn_executor_req, fut)
        self._in_queue.put(gnn_executor_req)

    def send_req_cgp(self, req_id, batch_id, target_features, sampling_rets, fut):
        target_features = target_features.to(self._device)

        cpu_global_in_degrees_list = []
        global_in_degrees_list = []
        all_reduce_futs = []
        for sampling_ret in sampling_rets:
            block_in_degrees = sampling_ret.block_in_degrees
            cpu_global_in_degrees_list.append(block_in_degrees)
            all_reduce_fut = dist.all_reduce(block_in_degrees, async_op=True, group=self._gloo_group)
            all_reduce_futs.append(all_reduce_fut)
            if self._full_sampling and self._use_precoms:
                break

        block_data_list, block_src_inputs_list = self._copy_block_data(sampling_rets)
        for i in range(len(all_reduce_futs)):
            all_reduce_futs[i].wait()
            global_in_degrees_list.append(cpu_global_in_degrees_list[i].to(self._device))

        gnn_executor_req = (batch_id, target_features, block_data_list, global_in_degrees_list, block_src_inputs_list)
        self._req_holder[batch_id] = (gnn_executor_req, fut)
        self._in_queue.put(gnn_executor_req)
    
    def _copy_block_data(self, sampling_rets):
        block_data_list = []
        block_src_inputs_list = []

        for sampling_ret in sampling_rets:
            block_src_ids = sampling_ret.block_src_ids
            block_src_ids = block_src_ids.to(self._device)
            unit_graph = sampling_ret.block_graph_idx.get_relation_graph(0).get_relation_graph(0)
            block_u, block_v, _ = unit_graph.edges(0)
            block_num_srcs = unit_graph.number_of_nodes(0)
            block_num_dsts = unit_graph.number_of_nodes(1)
            block_u = block_u.to(self._device)
            block_v = block_v.to(self._device)

            block_data_list.append((
                block_u,
                block_v,
                block_src_ids,
                block_num_srcs,
                block_num_dsts,
            ))

            if self._full_sampling and self._use_precoms:
                break

        for sampling_ret in sampling_rets:
            block_src_inputs = sampling_ret.block_src_inputs
            block_src_inputs = block_src_inputs.to(self._device)
            block_src_inputs_list.append(block_src_inputs)
        return block_data_list, block_src_inputs_list

    def shutdown(self):
        self._in_queue.put((None, None, None, None, None))
        self._callback_thread.join()
        self._finish_event.set()
        self._gnn_executor_process.join()
        assert len(self._req_holder) == 0

    @staticmethod
    def callback_thread_main(req_holder, out_queue, device, wait_sem):
        with torch.no_grad():
            while True:
                batch_id, ret_tensor = out_queue.get()
                if batch_id is None:
                    break

                ret_tensor_cpu = ret_tensor.cpu()
                del ret_tensor

                wait_sem.release()
                gnn_executor_req = req_holder[batch_id]
                del req_holder[batch_id]
                gnn_executor_req[-1].set_result((batch_id, ret_tensor_cpu))

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
        world_size=world_size * num_omega_groups + 1 + num_machines
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
