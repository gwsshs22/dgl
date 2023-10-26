import os
import sys
import time

import torch

import dgl

from dgl.omega.trace import trace_me, put_trace

from .distributed_block import DGLDistributedBlock
from .omega_apis import get_num_assigned_targets_per_gpu
from ..heterograph import DGLBlock
from .. import backend as F
from .. import utils
from .._ffi.function import _init_api
from .._ffi.object import ObjectBase, register_object

empty_graph = dgl.graph(([], []))

@register_object("omega.sampler_v2.SamplingExecutorV2")
class SamplingExecutorV2(ObjectBase):

    def __new__(cls):
        obj = ObjectBase.__new__(cls)
        obj._cache = {}
        return obj

    def sample_blocks_dp(self, batch_id, target_gnids, src_gnids, dst_gnids):
        ret_list = _CAPI_DGLOmegaSampleBlocksDp(
            self,
            batch_id,
            F.zerocopy_to_dgl_ndarray(target_gnids),
            F.zerocopy_to_dgl_ndarray(src_gnids),
            F.zerocopy_to_dgl_ndarray(dst_gnids)
        )

        graph_idx_list = []
        src_ids_list = []
        
        num_layers = len(ret_list) // 3
        for i in range(num_layers):
            graph_idx_list.append(ret_list[2 * i])
            src_ids_list.append(F.zerocopy_from_dgl_ndarray(ret_list[2 * i + 1]))
            
        fetched_inputs_list = [F.zerocopy_from_dgl_ndarray(r) for r in ret_list[-num_layers:]]
        return graph_idx_list, src_ids_list, fetched_inputs_list

    def sample_blocks_precoms(self, batch_id, target_gnids, src_gnids, dst_gnids, new_lhs_ids_prefix):
        ret_list = _CAPI_DGLOmegaSampleBlocksPrecoms(
            self,
            batch_id,
            F.zerocopy_to_dgl_ndarray(target_gnids),
            F.zerocopy_to_dgl_ndarray(src_gnids),
            F.zerocopy_to_dgl_ndarray(dst_gnids),
            F.zerocopy_to_dgl_ndarray(new_lhs_ids_prefix)
        )

        graph_idx_list = []
        src_ids_list = []
        
        num_layers = len(ret_list) // 3
        for i in range(num_layers):
            graph_idx_list.append(ret_list[2 * i])
            src_ids_list.append(F.zerocopy_from_dgl_ndarray(ret_list[2 * i + 1]))
            
        fetched_inputs_list = [F.zerocopy_from_dgl_ndarray(r) for r in ret_list[-num_layers:]]
        return graph_idx_list, src_ids_list, fetched_inputs_list

    def sample_blocks_dp_precoms_with_recom(self, batch_id, target_gnids, src_gnids, dst_gnids):
        ret_list = _CAPI_DGLOmegaSampleBlocksDpPrecomsWithRecom(
            self,
            batch_id,
            F.zerocopy_to_dgl_ndarray(target_gnids),
            F.zerocopy_to_dgl_ndarray(src_gnids),
            F.zerocopy_to_dgl_ndarray(dst_gnids))

        graph_idx_list = [ret_list[0], ret_list[2]]
        src_ids_list = [F.zerocopy_from_dgl_ndarray(ret_list[1]), F.zerocopy_from_dgl_ndarray(ret_list[3])]
        fetched_inputs_list = [F.zerocopy_from_dgl_ndarray(r) for r in ret_list[4:-1]]
        recompute_mask = F.zerocopy_from_dgl_ndarray(ret_list[-1])

        return graph_idx_list, src_ids_list, fetched_inputs_list, recompute_mask

    def sample_blocks_cgp_with_recom(self, batch_id, target_gnids, src_gnids, dst_gnids, new_lhs_ids_prefix):
        ret_list = _CAPI_DGLOmegaSampleBlocksCgpWithRecom(
            self,
            batch_id,
            F.zerocopy_to_dgl_ndarray(target_gnids),
            F.zerocopy_to_dgl_ndarray(src_gnids),
            F.zerocopy_to_dgl_ndarray(dst_gnids),
            F.zerocopy_to_dgl_ndarray(new_lhs_ids_prefix))

        graph_idx_list = [ret_list[0], ret_list[2]]
        src_ids_list = [F.zerocopy_from_dgl_ndarray(ret_list[1]), F.zerocopy_from_dgl_ndarray(ret_list[3])]
        fetched_inputs_list = [F.zerocopy_from_dgl_ndarray(r) for r in ret_list[4:-2]]
        recompute_mask = F.zerocopy_from_dgl_ndarray(ret_list[-2])
        recom_block_num_assigned_target_nodes = F.zerocopy_from_dgl_ndarray(ret_list[-1]).tolist()

        return graph_idx_list, src_ids_list, fetched_inputs_list, recompute_mask, recom_block_num_assigned_target_nodes


def create_sampler(
    local_g,
    num_machines,
    machine_rank,
    num_gpus_per_machine_in_group,
    gpu_rank_in_group,
    local_rank,
    nid_partitions,
    num_layers,
    fanouts,
    is_cgp,
    recom_threshold,
    pull_fn,
    dist_sampling_fn,
    pe_recom_policy_fn,
    all_gather_fn,
    dist_edges_fn,
    local_data_store,
    in_degrees,
    out_degrees,
    gnid_to_local_id_mapping
):
    local_data_names = []
    local_data_tensors = []

    for k, v in local_data_store.items():
        local_data_names.append(k)
        local_data_tensors.append(F.zerocopy_to_dgl_ndarray(v))

    local_graph_idx = local_g._graph if local_g else empty_graph._graph
    local_grph_nid = local_g.ndata[dgl.NID] if local_g else torch.tensor([], dtype=torch.int64)

    return _CAPI_DGLOmegaCreateSampler(
            num_machines,
            machine_rank,
            num_gpus_per_machine_in_group,
            gpu_rank_in_group,
            local_rank,
            F.zerocopy_to_dgl_ndarray(torch.tensor(nid_partitions, dtype=torch.int64)),
            num_layers,
            F.zerocopy_to_dgl_ndarray(torch.tensor(fanouts, dtype=torch.int64)),
            is_cgp,
            recom_threshold,
            pull_fn,
            dist_sampling_fn,
            pe_recom_policy_fn,
            all_gather_fn,
            dist_edges_fn,
            empty_graph._graph,
            local_graph_idx,
            F.zerocopy_to_dgl_ndarray(local_grph_nid),
            local_data_names,
            local_data_tensors,
            F.zerocopy_to_dgl_ndarray(in_degrees),
            F.zerocopy_to_dgl_ndarray(out_degrees),
            F.zerocopy_to_dgl_ndarray(gnid_to_local_id_mapping))


_init_api("dgl.omega.sampler_v2")
