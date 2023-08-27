import os
import sys

import torch

import dgl

from .distributed_block import DGLDistributedBlock
from .omega_apis import get_num_assigned_targets_per_gpu
from ..heterograph import DGLBlock
from .. import backend as F
from .. import utils
from .._ffi.function import _init_api
from .._ffi.object import ObjectBase, register_object

empty_graph = dgl.graph(([], []))

@register_object("omega.sampler.SamplingExecutor")
class SamplingExecutor(ObjectBase):
    
    def __new__(cls):
        obj = ObjectBase.__new__(cls)
        obj._cache = {}
        return obj

    def enqueue(self, batch_id, target_gnids, src_gnids, dst_gnids, callback):
        _CAPI_DGLEnqueue(
            self,
            batch_id,
            F.zerocopy_to_dgl_ndarray(target_gnids),
            F.zerocopy_to_dgl_ndarray(src_gnids),
            F.zerocopy_to_dgl_ndarray(dst_gnids),
            callback)

    def shutdown(self):
        _CAPI_DGLShutdown(self)

class SamplerPool:

    def __init__(
        self,
        num_machines,
        machine_rank,
        num_gpus_per_machine_in_group,
        gpu_ranks,
        local_gpu_rank_in_group,
        exec_mode,
        use_precoms,
        num_layers,
        pull_fn,
        dist_sampling_fn,
        executor):
        self._num_machines = num_machines
        self._machine_rank = machine_rank
        self._num_gpus_per_machine_in_group = num_gpus_per_machine_in_group
        self._gpu_ranks = gpu_ranks
        self._local_gpu_rank_in_group = local_gpu_rank_in_group
        self._exec_mode = exec_mode
        self._use_precoms = use_precoms
        self._num_layers = num_layers
        self._pull_fn = pull_fn
        self._dist_sampling_fn = dist_sampling_fn
        self._executor = executor

        self._gpu_rank_in_group = self._num_gpus_per_machine_in_group * self._machine_rank + self._local_gpu_rank_in_group
        self._callback_holder = set()

    def enqueue(
        self,
        batch_id,
        target_gnids,
        src_gnids,
        dst_gnids,
        cont):

        def callback(ret_list):
            blocks = []
            src_inputs_list = []
            if self._exec_mode == "dp":
                for layer_idx in range(self._num_layers):
                    block = DGLBlock(ret_list[3 * layer_idx], (["_N"], ["_N"]), ["_E"])
                    block.srcdata[dgl.NID] = F.zerocopy_from_dgl_ndarray(ret_list[3 * layer_idx + 1])
                    blocks.append(block)
                    src_inputs_list.append(F.zerocopy_from_dgl_ndarray(ret_list[3 * layer_idx + 2]))
            else:
                num_assigned_target_nodes = get_num_assigned_targets_per_gpu(
                    self._num_machines,
                    self._num_gpus_per_machine_in_group,
                    target_gnids.shape[0])
                gpu_rank_in_group = self._machine_rank * self._num_gpus_per_machine_in_group + self._local_gpu_rank_in_group
                for layer_idx in range(self._num_layers):
                    block = DGLDistributedBlock(
                        gpu_rank_in_group,
                        self._gpu_ranks,
                        num_assigned_target_nodes,
                        gidx=ret_list[3 * layer_idx],
                        ntypes=(['_N'], ['_N']),
                        etypes=['_E'])
                    block.srcdata[dgl.NID] = F.zerocopy_from_dgl_ndarray(ret_list[3 * layer_idx + 1])
                    blocks.append(block)
                    src_inputs_list.append(F.zerocopy_from_dgl_ndarray(ret_list[3 * layer_idx + 2]))

            cont(blocks, src_inputs_list)
            self._callback_holder.remove(callback)
        
        self._callback_holder.add(callback)

        self._executor.enqueue(
            batch_id,
            target_gnids,
            src_gnids,
            dst_gnids,
            callback)
    
    def shutdown(self):
        self._executor.shutdown()


def create_sampler_pool(
    num_threads,
    num_machines,
    machine_rank,
    num_gpus_per_machine_in_group,
    gpu_ranks,
    local_gpu_rank_in_group,
    nid_partitions,
    exec_mode,
    use_precoms,
    num_layers,
    fanouts,
    local_g,
    local_data_store,
    pull_fn,
    dist_sampling_fn):

    local_data_names = []
    local_data_tensors = []

    for k, v in local_data_store.items():
        local_data_names.append(k)
        local_data_tensors.append(F.zerocopy_to_dgl_ndarray(v))

    return SamplerPool(
        num_machines,
        machine_rank,
        num_gpus_per_machine_in_group,
        gpu_ranks,
        local_gpu_rank_in_group,
        exec_mode,
        use_precoms,
        num_layers,
        pull_fn,
        dist_sampling_fn,
        _CAPI_DGLCreateSamplingExecutor(
            num_threads,
            num_machines,
            machine_rank,
            num_gpus_per_machine_in_group,
            local_gpu_rank_in_group,
            F.zerocopy_to_dgl_ndarray(torch.tensor(nid_partitions, dtype=torch.int64)),
            exec_mode,
            use_precoms,
            num_layers,
            F.zerocopy_to_dgl_ndarray(torch.tensor(fanouts, dtype=torch.int64)),
            pull_fn,
            dist_sampling_fn,
            empty_graph._graph,
            local_g._graph,
            F.zerocopy_to_dgl_ndarray(local_g.ndata[dgl.NID]),
            local_data_names,
            local_data_tensors)
    )

_init_api("dgl.omega.sampler")
