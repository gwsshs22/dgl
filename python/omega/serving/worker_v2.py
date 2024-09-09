import argparse
from dataclasses import dataclass
import heapq
import os
import sys
import threading
import queue
import json
import time
import gc
import traceback
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
    sample_edges,
    get_num_assigned_targets_per_gpu,
    to_block,
    to_distributed_block_v2,
)
from dgl.omega.trace import trace_me, get_traces, get_cpp_traces, enable_tracing, put_trace, trace_blocks, collect_dist_block_stats

import dgl.backend as F

from omega.models import create_model
from omega.serving.block_sampler import BlockSampler

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
        feature_cache_size,
        use_precoms,
        recom_threshold,
        recom_policy,
        model_config,
        random_seed,
        profiling,
        tracing):

        self._global_rank = global_rank

        if tracing:
            enable_tracing()

        fanouts = [int(fanout) for fanout in model_config.fanouts.split(",")]
        fanouts = [-1 if f == 0 else f for f in fanouts]

        self._req_queue = queue.SimpleQueue()
        self._process_thread = threading.Thread(
            target=process_main,
            args=(
                self._req_queue,
                master_ip,
                master_dist_comm_port,
                ip_config,
                net_type,
                num_machines,
                num_gpus_per_machine,
                part_config,
                omega_group_id,
                graph_name,
                machine_rank,
                global_rank,
                local_rank,
                exec_mode,
                feature_cache_size,
                use_precoms,
                recom_threshold,
                recom_policy,
                model_config,
                random_seed,
                profiling,
                tracing))
        self._process_thread.start()

    @rpc.functions.async_execution
    def process(self, req_id, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        fut = torch.futures.Future()

        self._req_queue.put((req_id, batch_id, target_gnids, target_features, src_gnids, dst_gnids, fut))

        return fut

    def collect_traces(self):
        python_traces = get_traces(f"worker_{self._global_rank}")
        cpp_traces = get_cpp_traces(f"worker_{self._global_rank}")

        return python_traces + cpp_traces

    def shutdown(self):
        self._req_queue.put((-1,))
        self._process_thread.join()

def process_main(
    req_queue,
    master_ip,
    master_dist_comm_port,
    ip_config,
    net_type,
    num_machines,
    num_gpus_per_machine,
    part_config,
    omega_group_id,
    graph_name,
    machine_rank,
    global_rank,
    local_rank,
    exec_mode,
    feature_cache_size,
    use_precoms,
    recom_threshold,
    recom_policy,
    model_config,
    random_seed,
    profiling,
    tracing):

    dgl.distributed.dist_context.initialize_omega_worker(
        ip_config,
        net_type,
        omega_group_id)

    load_dgl_graph = (not use_precoms) or recom_threshold < 100

    dist_g = dgl.distributed.DistGraph(graph_name, part_config=part_config, load_dgl_graph=load_dgl_graph)
    dist_g.barrier()

    gpu_ranks = None
    gloo_group = None
    global_nccl_group = None
    nccl_group = None
    if exec_mode == "cgp" or exec_mode == "cgp-multi":
        world_size = num_machines * num_gpus_per_machine
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_ip}:{master_dist_comm_port}",
            world_size=world_size,
            rank=global_rank)

        if exec_mode == "cgp":
            gpu_ranks = [r for r in range(world_size)]
            nccl_group = torch.distributed.new_group()
            global_nccl_group = nccl_group
            set_nccl_group(nccl_group)
            gloo_group = torch.distributed.new_group(backend="gloo")
        else:
            global_nccl_group = torch.distributed.new_group()
            for i in range(num_gpus_per_machine):
                ranks_in_group = [r for r in range(i, world_size + i, num_gpus_per_machine)]
                tmp_nccl_group = torch.distributed.new_group(ranks=ranks_in_group)
                tmp_gloo_group = torch.distributed.new_group(ranks=ranks_in_group, backend="gloo")
                if i == local_rank:
                    gpu_ranks = ranks_in_group
                    nccl_group = tmp_nccl_group
                    set_nccl_group(nccl_group)
                    gloo_group = tmp_gloo_group

        torch.distributed.barrier()

    exec_context = LocalExecutionContext(
        dist_g,
        gloo_group,
        nccl_group,
        global_nccl_group,
        graph_name,
        load_dgl_graph,
        part_config,
        num_machines,
        num_gpus_per_machine,
        machine_rank,
        global_rank,
        local_rank,
        gpu_ranks,
        exec_mode,
        feature_cache_size,
        use_precoms,
        recom_threshold,
        recom_policy,
        model_config,
        random_seed,
        tracing)

    def run():
        pending_requests = {}
        req_id_heap = []
        expected_req_id = 0

        with torch.no_grad():
            while True:
                request = req_queue.get()
                req_id = request[0]
                if req_id < 0:
                    break

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
        dist_g,
        gloo_group,
        nccl_group,
        global_nccl_group,
        graph_name,
        load_dgl_graph,
        part_config,
        num_machines,
        num_gpus_per_machine,
        machine_rank,
        global_rank,
        local_rank,
        gpu_ranks,
        exec_mode,
        feature_cache_size,
        use_precoms,
        recom_threshold,
        recom_policy,
        model_config,
        random_seed,
        tracing):
        self._dist_g = dist_g
        self._local_g = dist_g.local_partition
        self._gpb = dist_g.get_partition_book()
        self._gloo_group = gloo_group
        self._nccl_group = nccl_group
        self._global_nccl_group = global_nccl_group
        self._num_machines = num_machines
        self._num_gpus_per_machine = num_gpus_per_machine
        self._machine_rank = machine_rank
        self._global_rank = global_rank
        self._local_rank = local_rank
        self._gpu_ranks = gpu_ranks
        self._exec_mode = exec_mode
        self._feature_cache_size = feature_cache_size
        self._use_precoms = use_precoms
        self._recom_threshold = recom_threshold
        self._recom_policy = recom_policy
        self._model_config = model_config
        self._num_layers = model_config.num_layers
        self._part_config = part_config

        fanouts = [int(fanout) for fanout in model_config.fanouts.split(",")]
        fanouts = [-1 if f == 0 else f for f in fanouts]
        self._fanouts = fanouts
        assert len(self._fanouts) == self._num_layers
        self._tracing = tracing

        self._device = f"cuda:{self._local_rank}"

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self._load_model()
        self._load_degrees()
        self._set_cgp_conf(load_dgl_graph)
        self._create_feature_cache()
        self._create_block_sampler()
        self._large_full_dp = "PYTORCH_NO_CUDA_MEMORY_CACHING" in os.environ

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

    def _load_degrees(self):
        degrees = dgl.data.load_tensors(str(Path(self._part_config).parent / "degrees.dgl"))
        self._in_degrees = degrees["in_degrees"]
        if "out_degrees" in degrees:
            self._out_degrees = degrees["out_degrees"]
        else:
            self._out_degrees = self._in_degrees

    def _set_cgp_conf(self, load_dgl_graph):
        world_size = self._num_machines * self._num_gpus_per_machine

        if self._exec_mode == "cgp-multi":
            num_gpus_per_machine_in_group = 1
            gpu_ranks = [r for r in range(self._local_rank, world_size + self._local_rank, self._num_gpus_per_machine)]
            local_gpu_rank_in_group = 0
        else:
            num_gpus_per_machine_in_group = self._num_gpus_per_machine
            gpu_ranks = [r for r in range(world_size)]
            local_gpu_rank_in_group = self._local_rank

        self._num_gpus_per_machine_in_group = num_gpus_per_machine_in_group
        self._gpu_ranks = gpu_ranks
        self._local_gpu_rank_in_group = local_gpu_rank_in_group
        self._gpu_rank_in_group = self._machine_rank * self._num_gpus_per_machine_in_group + self._local_gpu_rank_in_group

        if self._fanouts[0] == -1:
            self._cgp_fanouts = self._fanouts
        else:
            
            self._cgp_fanouts = [
                max(1, get_num_assigned_targets_per_gpu(
                    self._num_machines, self._num_gpus_per_machine_in_group, f
                )[self._gpu_rank_in_group]) for f in self._fanouts
            ]

        if load_dgl_graph:
            self._gnid_to_local_id_mapping = torch.ones(self._dist_g.num_nodes(), dtype=torch.long) * -1
            self._gnid_to_local_id_mapping[self._local_g.ndata[dgl.NID]] = torch.arange(self._local_g.num_nodes())
        else:
            self._gnid_to_local_id_mapping = torch.tensor([])

    def _create_feature_cache(self):
        if self._feature_cache_size is None:
            self._feature_cache = None
            self._cached_id_map = torch.tensor([])
            return

        self._feature_cache = torch.empty(self._feature_cache_size, self._model_config.num_inputs, device=self._device, dtype=torch.float32)
        self._cached_id_map = torch.ones(self._out_degrees.shape[0], dtype=torch.int64) * -1

        max_node_ids = self._gpb._max_node_ids
        my_partid = self._gpb.partid
        num_partitions = self._gpb.num_partitions()
        out_degrees = self._out_degrees.clone()

        if self._exec_mode == "dp":
            start_idx = 0
            for p in range(num_partitions):
                end_idx = max_node_ids[p]
                if p == my_partid:
                    out_degrees[start_idx:end_idx] = -1
                    break
                start_idx = end_idx
        else:
            start_idx = 0
            for p in range(num_partitions):
                end_idx = max_node_ids[p]
                if p != my_partid:
                    out_degrees[start_idx:end_idx] = -1
                start_idx = end_idx
        assert (out_degrees >= 0).sum() >= self._feature_cache_size
        self._cached_id_map[out_degrees.sort().indices[-self._feature_cache_size:]] = torch.arange(self._feature_cache_size)
        assert (self._cached_id_map >= 0).sum() == self._feature_cache_size

        print(f"Cache size = {self._feature_cache.element_size() * self._feature_cache.nelement() / 1000000000 :.2f}GB")
        # self._cached_id_map[self._out_degrees.sort().indices[-self._feature_cache_size:]] = torch.arange(self._feature_cache_size)

    def _create_block_sampler(self):
        is_cgp = self._exec_mode == "cgp" or self._exec_mode == "cgp-multi"
        fanouts = self._cgp_fanouts if is_cgp else self._fanouts
        self._block_sampler = BlockSampler(
            self._dist_g,
            self._part_config,
            self._gloo_group,
            self._global_nccl_group,
            self._nccl_group,
            self._num_machines,
            self._machine_rank,
            self._num_gpus_per_machine_in_group,
            self._gpu_ranks,
            self._local_rank,
            self._gpu_rank_in_group,
            is_cgp,
            self._use_precoms,
            self._recom_threshold,
            self._num_layers,
            self._in_degrees,
            self._out_degrees,
            self._cached_id_map,
            self._gnid_to_local_id_mapping,
            fanouts,
            self._tracing,
            self._device
        )

    def in_degrees(self, global_nids):
        return self._in_degrees[global_nids]

    def out_degrees(self, global_nids):
        return self._out_degrees[global_nids]
    
    def local_in_edges(self, gnids):
        local_ids = self._gnid_to_local_id_mapping[gnids]
        local_ids = local_ids[local_ids >= 0]
        u, v = self._local_g.in_edges(local_ids)
        return self._local_g.ndata[dgl.NID][u], self._local_g.ndata[dgl.NID][v]

    def process_request(self, req_id, batch_id, target_gnids, target_features, src_gnids, dst_gnids, fut):
        try:
            if self._use_precoms:
                if self._exec_mode == "dp":
                    ret_tensor = self.execute_dp_with_precoms(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
                elif self._exec_mode == "cgp" or self._exec_mode == "cgp-multi":
                    ret_tensor = self.execute_cgp_with_precoms(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
            else:
                if self._exec_mode == "dp":
                    if self._large_full_dp:
                        ret_tensor = self.execute_large_full_dp(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
                    else:    
                        ret_tensor = self.execute_dp(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
                elif self._exec_mode == "cgp" or self._exec_mode == "cgp-multi":
                    raise NotImplementedError()
        except Exception as ex:
            traceback.print_exc()
            fut.set_exception(ex)
            return

        fut.set_result((batch_id, ret_tensor))

    def execute_large_full_dp(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        with trace_me(batch_id, "copy", self._device):
            src_gnids = src_gnids.to(self._device)
            dst_gnids = dst_gnids.to(self._device)
            target_gnids = target_gnids.to(self._device)

        with trace_me(batch_id, "mfg", self._device):
            blocks, fetched_inputs_list = self._block_sampler.sample_blocks_dp(batch_id, target_gnids, src_gnids, dst_gnids)
            trace_blocks(batch_id, blocks)

        with trace_me(batch_id, "copy", self._device):
            input_features = fetched_inputs_list[0]
            input_features = self._copy_features(input_features, blocks[0].srcdata[dgl.NID][target_gnids.shape[0]:])
            target_features = target_features.to(self._device)
            features = torch.concat((
                target_features,
                input_features
            ))

        with trace_me(batch_id, "delete", self._device):
            del fetched_inputs_list
            del input_features
            gc.collect()

        with trace_me(batch_id, "compute", self._device):
            h0 = self._model.feature_preprocess(features)

            h = h0
            h = self._model.layer_foward(0, blocks[0], h, h0)

        with trace_me(batch_id, "delete", self._device):
            blocks.pop(0)
            del features
            gc.collect()
    
        for layer_idx in range(1, self._num_layers):
            with trace_me(batch_id, "compute", self._device):
                h = self._model.layer_foward(layer_idx, blocks[0], h, h0)

            with trace_me(batch_id, "delete", self._device):
                blocks.pop(0)
                gc.collect()

        with trace_me(batch_id, "compute", self._device):
            h = h.cpu()
    
        with trace_me(batch_id, "delete", self._device):
            del blocks

        return h

    def execute_dp(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        with trace_me(batch_id, "copy", self._device):
            src_gnids = src_gnids.to(self._device)
            dst_gnids = dst_gnids.to(self._device)
            target_gnids = target_gnids.to(self._device)

        with trace_me(batch_id, "mfg", self._device):
            blocks, fetched_inputs_list = self._block_sampler.sample_blocks_dp(batch_id, target_gnids, src_gnids, dst_gnids)
            trace_blocks(batch_id, blocks)

        with trace_me(batch_id, "copy", self._device):
            input_features = fetched_inputs_list[0]
            input_features = self._copy_features(input_features, blocks[0].srcdata[dgl.NID][target_gnids.shape[0]:])
            target_features = target_features.to(self._device)
            features = torch.concat((
                target_features,
                input_features
            ))

        with trace_me(batch_id, "compute", self._device):
            h = self._model(blocks, features)
            h = h.cpu()

        with trace_me(batch_id, "delete", self._device):
            del fetched_inputs_list
            del blocks
            del input_features
            del features
            del target_features

        return h

    def execute_dp_with_precoms(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        if self._recom_threshold == 100:
            return self.execute_dp_with_precoms_no_recom(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
        else:
            return self.execute_dp_with_precoms_with_recom(batch_id, target_gnids, target_features, src_gnids, dst_gnids)

    def execute_dp_with_precoms_no_recom(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        with trace_me(batch_id, "copy", self._device):
            src_gnids = src_gnids.to(self._device)
            dst_gnids = dst_gnids.to(self._device)
            target_gnids = target_gnids.to(self._device)

        with trace_me(batch_id, "mfg", self._device):
            blocks, fetched_inputs_list = self._block_sampler.sample_blocks_dp_precoms(batch_id, target_gnids, src_gnids, dst_gnids)
            trace_blocks(batch_id, blocks)

        with trace_me(batch_id, "copy", self._device):
            input_features = fetched_inputs_list[0]
            input_features = self._copy_features(input_features, blocks[0].srcdata[dgl.NID][target_gnids.shape[0]:])
            inputs_list = [l.to(self._device) for l in fetched_inputs_list[1:]]
            inputs_list.insert(0, input_features)
            features = torch.concat((
                target_features.to(self._device),
                inputs_list[0]
            ))

        with trace_me(batch_id, "compute", self._device):
            h0 = self._model.feature_preprocess(features)

            h = h0
            h = self._model.layer_foward(0, blocks[0], h, h0)

            for layer_idx in range(1, self._num_layers):
                h = torch.concat((h, inputs_list[layer_idx]))
                h = self._model.layer_foward(layer_idx, blocks[layer_idx], h, h0)

            h = h.cpu()

        with trace_me(batch_id, "delete", self._device):
            del fetched_inputs_list
            del blocks

        return h

    def execute_dp_with_precoms_with_recom(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        with trace_me(batch_id, "copy", self._device):
            # Support recom only with full sampling
            assert self._fanouts[0] == -1
            assert 0 <= self._recom_threshold and self._recom_threshold < 100
            batch_size = target_gnids.shape[0]
            target_gnids = target_gnids.to(self._device)
            src_gnids = src_gnids.to(self._device)
            dst_gnids = dst_gnids.to(self._device)

        with trace_me(batch_id, "mfg", self._device):
            recompute_block, recompute_mask, block, fetched_inputs_list = self._block_sampler.sample_blocks_dp_precoms_with_recom(
                batch_id,
                target_gnids,
                src_gnids,
                dst_gnids)
            trace_blocks(batch_id, [recompute_block, block])

        with trace_me(batch_id, "copy", self._device):
            input_features = fetched_inputs_list[0]
            input_features = self._copy_features(input_features, recompute_block.srcdata[dgl.NID][target_gnids.shape[0]:])
            inputs_list = [l.to(self._device) for l in fetched_inputs_list[1:]]
            inputs_list.insert(0, input_features)
            h = torch.concat((
                target_features.to(self._device),
                inputs_list[0]))


        with trace_me(batch_id, "compute", self._device):
            h0 = self._model.feature_preprocess(h)
            h = h0    
            h = self._model.layer_foward(0, recompute_block, h, h0)

            for layer_idx in range(1, self._num_layers - 1):
                h = torch.concat((h, inputs_list[layer_idx]))
                h = self._model.layer_foward(layer_idx, recompute_block, h, h0)

            pe = torch.zeros((block.num_src_nodes() - block.num_dst_nodes(),) + h.shape[1:], device=self._device)
            pe[recompute_mask] = h[batch_size:]
            pe[torch.logical_not(recompute_mask)] =  inputs_list[-1]

            h = torch.concat((
                h[:batch_size],
                pe
            ))

            h = self._model.layer_foward(self._num_layers - 1, block, h, h0)
            h = h.cpu()

        with trace_me(batch_id, "delete", self._device):
            del fetched_inputs_list
            del recompute_block
            del block

        return h

    def execute_cgp_with_precoms(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        if self._recom_threshold == 100:
            return self.execute_cgp_with_precoms_no_recom(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
        else:
            return self.execute_cgp_with_precoms_with_recom(batch_id, target_gnids, target_features, src_gnids, dst_gnids)

    def execute_cgp_with_precoms_no_recom(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        with trace_me(batch_id, "copy", self._device):
            src_gnids = src_gnids.to(self._device)
            dst_gnids = dst_gnids.to(self._device)
            target_gnids = target_gnids.to(self._device)

        with trace_me(batch_id, "mfg", self._device):
            blocks, fetched_inputs_list = self._block_sampler.sample_blocks_cgp(batch_id, target_gnids, src_gnids, dst_gnids)
            trace_blocks(batch_id, blocks)

        with trace_me(batch_id, "copy", self._device):
            input_features = fetched_inputs_list[0]
            input_features = self._copy_features(input_features, blocks[0].srcdata[dgl.NID][target_features.shape[0]:])
            inputs_list = [l.to(self._device) for l in fetched_inputs_list[1:]]
            inputs_list.insert(0, input_features)
            features = torch.concat((
                target_features.to(self._device),
                inputs_list[0]
            ))

        with trace_me(batch_id, "compute", self._device):
            h0 = self._model.feature_preprocess(features)

            h = h0
            h = self._model.layer_foward(0, blocks[0], h, h0)

            for layer_idx in range(1, self._num_layers):
                h = torch.concat((h, inputs_list[layer_idx]))
                h = self._model.layer_foward(layer_idx, blocks[layer_idx], h, h0)
            h = h.cpu()

        with trace_me(batch_id, "delete", self._device):
            collect_dist_block_stats(batch_id, blocks)
            del fetched_inputs_list
            del blocks

        return h

    def execute_cgp_with_precoms_with_recom(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        with trace_me(batch_id, "copy", self._device):
            # Support recom only with full sampling
            assert self._fanouts[0] == -1
            assert 0 <= self._recom_threshold and self._recom_threshold < 100

            target_gnids = target_gnids.to(self._device)
            src_gnids = src_gnids.to(self._device)
            dst_gnids = dst_gnids.to(self._device)

        with trace_me(batch_id, "mfg", self._device):
            recompute_block, recompute_mask, block, fetched_inputs_list, num_local_target_nodes = self._block_sampler.sample_blocks_cgp_with_recom(
                batch_id,
                target_gnids,
                src_gnids,
                dst_gnids)
            blocks = [recompute_block, block]
            trace_blocks(batch_id, blocks)

        with trace_me(batch_id, "copy", self._device):
            input_features = fetched_inputs_list[0]
            input_features = self._copy_features(input_features, blocks[0].srcdata[dgl.NID][target_features.shape[0]:])
            inputs_list = [l.to(self._device) for l in fetched_inputs_list[1:]]
            inputs_list.insert(0, input_features)
            h = torch.concat((
                target_features.to(self._device),
                inputs_list[0]))

        with trace_me(batch_id, "compute", self._device):
            h0 = self._model.feature_preprocess(h)
            h = h0    
            h = self._model.layer_foward(0, recompute_block, h, h0)

            for layer_idx in range(1, self._num_layers - 1):
                h = torch.concat((h, inputs_list[layer_idx]))
                h = self._model.layer_foward(layer_idx, recompute_block, h, h0)

            pe = torch.zeros((block.num_src_nodes() - block.num_dst_nodes(),) + h.shape[1:], device=self._device)
            pe[recompute_mask] = h[num_local_target_nodes:]
            pe[torch.logical_not(recompute_mask)] =  inputs_list[-1]

            h = torch.concat((
                h[:num_local_target_nodes],
                pe
            ))

            h = self._model.layer_foward(self._num_layers - 1, block, h, h0)
            h = h.cpu()

        with trace_me(batch_id, "delete", self._device):
            collect_dist_block_stats(batch_id, blocks)
            del fetched_inputs_list
            del blocks
            del recompute_block
            del block

        return h

    def _copy_features(self, fetched_features, src_ids):
        fetched_features = fetched_features.to(self._device)
        if self._feature_cache_size is None:
            return fetched_features

        input_features = torch.empty(src_ids.shape[0], fetched_features.shape[1], dtype=fetched_features.dtype, device=self._device)
        cache_slots = self._cached_id_map[src_ids.to("cpu")].to(self._device)
        cached_index = cache_slots >= 0

        input_features[torch.logical_not(cached_index)] = fetched_features
        input_features[cached_index] = self._feature_cache[cache_slots[cached_index]]

        return input_features

    def _all_gather_recompute_block_target_ids(self, local_recompute_block_target_ids):
        group_size = len(self._gpu_ranks)
        sizes = torch.zeros(group_size, dtype=torch.long)
        sizes[self._gpu_rank_in_group] = local_recompute_block_target_ids.shape[0]
        local_recompute_block_target_ids = local_recompute_block_target_ids.to("cpu")
        sizes_list = list(sizes.split([1] * group_size))
        torch.distributed.all_gather(sizes_list, sizes_list[self._gpu_rank_in_group], group=self._gloo_group)

        recompute_block_target_ids = torch.empty(sizes.sum(), dtype=torch.long)

        sizes = sizes.tolist()
        outputs = list(recompute_block_target_ids.split(sizes))
        req_handles = []
        for i in range(group_size):
            if i == self._gpu_rank_in_group:
                req_handles.append(torch.distributed.broadcast(local_recompute_block_target_ids, self._gpu_ranks[i], async_op=True, group=self._gloo_group))
            else:
                req_handles.append(torch.distributed.broadcast(outputs[i], self._gpu_ranks[i], async_op=True, group=self._gloo_group))

        outputs[self._gpu_rank_in_group].copy_(local_recompute_block_target_ids)

        for r in req_handles:
            r.wait()

        return recompute_block_target_ids, sizes

    def _distribute_edges(self, batch_id, target_gnids, num_assigned_target_nodes, u, v):
        group_size = len(self._gpu_ranks)

        u_list = []
        v_list = []
        sizes = []
        off = 0
        for i in range(group_size):
            start_id = target_gnids[off]
            end_id = target_gnids[off + num_assigned_target_nodes[i] - 1]
            mask = torch.logical_and(u >= start_id, u <= end_id)
            u_list.append(u[mask])
            v_list.append(v[mask])
            sizes.append(u_list[-1].shape[0])
            off += num_assigned_target_nodes[i]

        assert sum(sizes) == u.shape[0]

        num_edges_per_partition = torch.tensor(sizes, dtype=torch.int64, device=self._device)

        all_num_edges_per_partition = torch.zeros((group_size * group_size), dtype=torch.int64, device=self._device)

        torch.distributed.all_gather(list(all_num_edges_per_partition.split([group_size] * group_size)), num_edges_per_partition, group=self._nccl_group)

        all_num_edges_per_partition = all_num_edges_per_partition.reshape(group_size, group_size)
        expected_num_per_partition = all_num_edges_per_partition.transpose(0, 1)[self._gpu_rank_in_group].tolist()

        u_output = []
        v_output = []
        for i in range(group_size):
            if i == self._gpu_rank_in_group:
                u_output.append(u_list[self._gpu_rank_in_group])
                v_output.append(v_list[self._gpu_rank_in_group])
            else:
                u_output.append(torch.zeros(expected_num_per_partition[i], dtype=torch.int64, device=self._device))
                v_output.append(torch.zeros(expected_num_per_partition[i], dtype=torch.int64, device=self._device))

        req_handles = []
        req_handles.append(torch.distributed.all_to_all(u_output, u_list, group=self._nccl_group, async_op=True))
        req_handles.append(torch.distributed.all_to_all(v_output, v_list, group=self._nccl_group, async_op=True))
        
        u = torch.concat(u_output)
        v = torch.concat(v_output)

        return u, v

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
