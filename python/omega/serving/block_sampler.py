import os
import sys
import json
import time
import gc
import traceback
import queue
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.distributed.rpc as rpc
import numpy as np

import dgl
from dgl.distributed.kvstore import get_kvstore
from dgl.omega.dist_context import set_nccl_group
from dgl.heterograph import DGLBlock

from dgl.omega.distributed_block import DGLDistributedBlock
from dgl.omega.omega_apis import get_num_assigned_targets_per_gpu
from dgl.omega import sampler_v2
from dgl.omega.trace import trace_me, get_traces, get_cpp_traces, enable_tracing, put_trace
from omega.serving.graph_server import OmegaGraphServer

import dgl.backend as F

class BlockSampler:
    def __init__(
        self,
        dist_g,
        part_config,
        gloo_group,
        global_nccl_group,
        nccl_group,
        num_machines,
        machine_rank,
        num_gpus_per_machine_in_group,
        gpu_ranks,
        local_rank,
        gpu_rank_in_group,
        is_cgp,
        use_precoms,
        recom_threshold,
        num_layers,
        in_degrees,
        out_degrees,
        cached_id_map,
        gnid_to_local_id_mapping,
        fanouts,
        tracing,
        device):

        self._dist_g = dist_g
        self._gloo_group = gloo_group
        self._global_nccl_group = global_nccl_group
        self._nccl_group = nccl_group
        self._gpu_ranks = gpu_ranks
        self._gpu_rank_in_group = gpu_rank_in_group
        self._recom_threshold = recom_threshold

        self._local_g = self._dist_g.local_partition
        self._local_data_store = {}
        self._in_degrees = in_degrees
        self._out_degrees = out_degrees
        self._cached_id_map = cached_id_map
        self._gnid_to_local_id_mapping = gnid_to_local_id_mapping
        self._device = device
        self._callback_holder = set()
        self._tracing = tracing

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
        self._num_gpus_per_machine_in_group = num_gpus_per_machine_in_group
        
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

        nid_partitions = []
        part_config = json.loads(Path(part_config).read_text())
        assert len(part_config["node_map"]) == 1, "Support only homogeneous graphs currently."
        assert "_N" in part_config["node_map"], "Support only homogeneous graphs currently."
        nid_splits = part_config["node_map"]["_N"]
        nid_partitions = [s[0] for s in nid_splits]
        nid_partitions.append(nid_splits[-1][1])

        self._sampler = sampler_v2.create_sampler(
            self._local_g,
            num_machines,
            machine_rank,
            num_gpus_per_machine_in_group,
            self._gpu_rank_in_group,
            local_rank,
            nid_partitions,
            num_layers,
            fanouts,
            is_cgp,
            recom_threshold,
            self.pull_fn,
            self.dist_sampling_fn,
            self.pe_recom_policy_fn,
            self.all_gather_fn,
            self.dist_edges_fn,
            self.filter_cached_id_fn,
            self._local_data_store,
            in_degrees,
            out_degrees,
            self._cached_id_map,
            gnid_to_local_id_mapping)

    def sample_blocks_dp(self, batch_id, target_gnids, src_gnids, dst_gnids):
        graph_idx_list, src_ids_list, fetched_inputs_list = self._sampler.sample_blocks_dp(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids)

        blocks = []
        for graph_idx, src_ids in zip(graph_idx_list, src_ids_list):
            block = DGLBlock(graph_idx, (["_N"], ["_N"]), ["_E"])
            block.srcdata[dgl.NID] = src_ids
            blocks.append(block)

        return blocks, fetched_inputs_list

    def sample_blocks_dp_precoms(self, batch_id, target_gnids, src_gnids, dst_gnids):
        graph_idx_list, src_ids_list, fetched_inputs_list = self._sampler.sample_blocks_precoms(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids,
            target_gnids)

        blocks = []
        for graph_idx, src_ids in zip(graph_idx_list, src_ids_list):
            block = DGLBlock(graph_idx, (["_N"], ["_N"]), ["_E"])
            block.srcdata[dgl.NID] = src_ids
            blocks.append(block)

        return blocks, fetched_inputs_list

    def sample_blocks_dp_precoms_with_recom(self, batch_id, target_gnids, src_gnids, dst_gnids):
        graph_idx_list, src_ids_list, fetched_inputs_list, recompute_mask = self._sampler.sample_blocks_dp_precoms_with_recom(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids)

        assert len(graph_idx_list) == 2
        recompute_block = DGLBlock(graph_idx_list[0], (["_N"], ["_N"]), ["_E"])
        recompute_block.srcdata[dgl.NID] = src_ids_list[0]

        block = DGLBlock(graph_idx_list[1], (["_N"], ["_N"]), ["_E"])
        block.srcdata[dgl.NID] = src_ids_list[1]

        return recompute_block, recompute_mask.type(torch.bool), block, fetched_inputs_list

    def sample_blocks_cgp(self, batch_id, target_gnids, src_gnids, dst_gnids):
        batch_size = target_gnids.shape[0]
        num_assigned_target_nodes = get_num_assigned_targets_per_gpu(
            self._num_machines,
            self._num_gpus_per_machine_in_group,
            batch_size)

        local_target_nodes_start_idx = 0
        for i in range(self._gpu_rank_in_group):
            local_target_nodes_start_idx += num_assigned_target_nodes[i]
        num_local_target_nodes = num_assigned_target_nodes[self._gpu_rank_in_group]
        new_lhs_ids_prefix = target_gnids[local_target_nodes_start_idx:local_target_nodes_start_idx + num_local_target_nodes]
        graph_idx_list, src_ids_list, fetched_inputs_list = self._sampler.sample_blocks_precoms(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids,
            new_lhs_ids_prefix)

        blocks = []
        for graph_idx, src_ids in zip(graph_idx_list, src_ids_list):
            dist_block = DGLDistributedBlock(
                self._gpu_rank_in_group,
                self._gpu_ranks,
                num_assigned_target_nodes,
                gidx=graph_idx,
                ntypes=(['_N'], ['_N']),
                etypes=['_E'],
                tracing=self._tracing)

            dist_block.srcdata[dgl.NID] = src_ids
            blocks.append(dist_block)

        return blocks, fetched_inputs_list

    def sample_blocks_cgp_with_recom(self, batch_id, target_gnids, src_gnids, dst_gnids):
        batch_size = target_gnids.shape[0]
        num_assigned_target_nodes = get_num_assigned_targets_per_gpu(
            self._num_machines,
            self._num_gpus_per_machine_in_group,
            batch_size)

        local_target_nodes_start_idx = 0
        for i in range(self._gpu_rank_in_group):
            local_target_nodes_start_idx += num_assigned_target_nodes[i]
        num_local_target_nodes = num_assigned_target_nodes[self._gpu_rank_in_group]
        new_lhs_ids_prefix = target_gnids[local_target_nodes_start_idx:local_target_nodes_start_idx + num_local_target_nodes]

        graph_idx_list, src_ids_list, fetched_inputs_list, recompute_mask, recom_block_num_assigned_target_nodes = self._sampler.sample_blocks_cgp_with_recom(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids,
            new_lhs_ids_prefix)

        assert len(graph_idx_list) == 2
        recompute_block = DGLDistributedBlock(
                self._gpu_rank_in_group,
                self._gpu_ranks,
                recom_block_num_assigned_target_nodes,
                gidx=graph_idx_list[0],
                ntypes=(['_N'], ['_N']),
                etypes=['_E'],
                tracing=self._tracing)
        recompute_block.srcdata[dgl.NID] = src_ids_list[0]

        block = DGLDistributedBlock(
                self._gpu_rank_in_group,
                self._gpu_ranks,
                num_assigned_target_nodes,
                gidx=graph_idx_list[1],
                ntypes=(['_N'], ['_N']),
                etypes=['_E'],
                tracing=self._tracing)
        block.srcdata[dgl.NID] = src_ids_list[1]

        return recompute_block, recompute_mask.type(torch.bool), block, fetched_inputs_list, num_local_target_nodes

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

    def pe_recom_policy_fn(self, inputs):
        direct_neighbor_nids = F.zerocopy_from_dgl_ndarray(inputs[0])
        new_degrees = F.zerocopy_from_dgl_ndarray(inputs[1])

        org_degrees = self._in_degrees[direct_neighbor_nids]
        metrics = new_degrees / (new_degrees + org_degrees)
        m_threshold = np.percentile(metrics, self._recom_threshold)
        recompute_mask = (metrics >= m_threshold)
        recompute_ids = direct_neighbor_nids[recompute_mask]
        reuse_ids = direct_neighbor_nids[torch.logical_not(recompute_mask)]

        return [F.zerocopy_to_dgl_ndarray(recompute_ids), \
            F.zerocopy_to_dgl_ndarray(reuse_ids), \
            F.zerocopy_to_dgl_ndarray(recompute_mask), \
            F.zerocopy_to_dgl_ndarray(recompute_mask.nonzero().reshape(-1))]

    def all_gather_fn(self, inputs):
        local_tensor = F.zerocopy_from_dgl_ndarray(inputs[0])
        group_size = len(self._gpu_ranks)
        sizes = torch.zeros(group_size, dtype=torch.long)
        sizes[self._gpu_rank_in_group] = local_tensor.shape[0]
        sizes_list = list(sizes.split([1] * group_size))
        torch.distributed.all_gather(sizes_list, sizes_list[self._gpu_rank_in_group], group=self._gloo_group)

        output = torch.empty(sizes.sum(), dtype=torch.long)

        outputs = list(output.split(sizes.tolist()))
        req_handles = []
        for i in range(group_size):
            if i == self._gpu_rank_in_group:
                req_handles.append(torch.distributed.broadcast(local_tensor, self._gpu_ranks[i], async_op=True, group=self._gloo_group))
            else:
                req_handles.append(torch.distributed.broadcast(outputs[i], self._gpu_ranks[i], async_op=True, group=self._gloo_group))

        outputs[self._gpu_rank_in_group].copy_(local_tensor)

        for r in req_handles:
            r.wait()

        return [F.zerocopy_to_dgl_ndarray(output), F.zerocopy_to_dgl_ndarray(sizes)]

    def dist_edges_fn(self, src_ids_list, dst_ids_list):
        src_ids_list = [F.zerocopy_from_dgl_ndarray(l) for l in src_ids_list]
        dst_ids_list = [F.zerocopy_from_dgl_ndarray(l) for l in dst_ids_list]
        sizes = [l.shape[0] for l in src_ids_list]
        group_size = len(src_ids_list)
        num_edges_per_partition = torch.tensor(sizes, dtype=torch.int64, device=self._device)

        all_num_edges_per_partition = torch.zeros((group_size * group_size), dtype=torch.int64, device=self._device)

        torch.distributed.all_gather(list(all_num_edges_per_partition.split([group_size] * group_size)), num_edges_per_partition, group=self._nccl_group)

        all_num_edges_per_partition = all_num_edges_per_partition.reshape(group_size, group_size)
        expected_num_per_partition = all_num_edges_per_partition.transpose(0, 1)[self._gpu_rank_in_group].tolist()

        u_output = []
        v_output = []
        for i in range(group_size):
            if i == self._gpu_rank_in_group:
                u_output.append(src_ids_list[self._gpu_rank_in_group])
                v_output.append(dst_ids_list[self._gpu_rank_in_group])
            else:
                u_output.append(torch.empty(expected_num_per_partition[i], dtype=torch.int64, device=self._device))
                v_output.append(torch.empty(expected_num_per_partition[i], dtype=torch.int64, device=self._device))

        req_handles = []
        req_handles.append(torch.distributed.all_to_all(u_output, src_ids_list, group=self._nccl_group, async_op=True))
        req_handles.append(torch.distributed.all_to_all(v_output, dst_ids_list, group=self._nccl_group, async_op=True))
        
        sss = time.time()
        for h in req_handles:
            h.wait()

        u = torch.concat(u_output)
        v = torch.concat(v_output)

        return [F.zerocopy_to_dgl_ndarray(u), F.zerocopy_to_dgl_ndarray(v)]
    
    def filter_cached_id_fn(self, id_arr):
        id_arr = F.zerocopy_from_dgl_ndarray(id_arr[0])
        id_arr = id_arr[self._cached_id_map[id_arr] < 0]
        return [F.zerocopy_to_dgl_ndarray(id_arr)]
