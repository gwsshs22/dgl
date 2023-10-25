import torch

import dgl

from .distributed_block import DGLDistributedBlock
from ..heterograph import DGLBlock
from .._ffi.function import _init_api
from .. import backend as F
from .. import utils

empty_graph = dgl.graph(([], []))
empty_tensor = torch.tensor([])

def get_num_assigned_targets_per_gpu(num_machines, num_gpus_per_machine, num_targets):
    num_total_gpus = num_machines * num_gpus_per_machine
    num_targets_per_gpu = [0] * num_total_gpus
    for machine_idx in range(num_machines):
        num_targets_in_machine = num_targets // num_machines
        if machine_idx < num_targets % num_machines:
            num_targets_in_machine += 1

        for gpu_idx in range(num_gpus_per_machine):
            global_gpu_idx = machine_idx * num_gpus_per_machine + gpu_idx
            num_targets_per_gpu[global_gpu_idx] = num_targets_in_machine // num_gpus_per_machine
            if gpu_idx < num_targets_in_machine % num_gpus_per_machine:
                num_targets_per_gpu[global_gpu_idx] += 1
    return num_targets_per_gpu

def to_distributed_blocks(
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    target_gnids,
    src_gnids,
    src_part_ids,
    dst_gnids):
    ret = _CAPI_DGLOmegaToDistributedBlocks(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        F.zerocopy_to_dgl_ndarray(target_gnids),
        F.zerocopy_to_dgl_ndarray(src_gnids),
        F.zerocopy_to_dgl_ndarray(src_part_ids),
        F.zerocopy_to_dgl_ndarray(dst_gnids))

    num_assigned_target_nodes = get_num_assigned_targets_per_gpu(
        num_machines, num_gpus_per_machine, target_gnids.shape[0])
    dist_blocks = []
    gpu_ranks = [i for i in range(num_machines * num_gpus_per_machine)]
    for gpu_idx in range(num_gpus_per_machine):
        g_idx = ret[2 * gpu_idx]
        src_gnids_in_block = F.from_dgl_nd(ret[2 * gpu_idx + 1])
        gpu_rank_in_group = num_gpus_per_machine * machine_rank + gpu_idx

        dist_block = DGLDistributedBlock(
            gpu_rank_in_group,
            gpu_ranks,
            num_assigned_target_nodes,
            gidx=g_idx,
            ntypes=(['_N'], ['_N']),
            etypes=['_E'])

        assert dist_block.is_unibipartite
        dist_block.srcdata[dgl.NID] = src_gnids_in_block
        dist_blocks.append(dist_block)
    return dist_blocks

def to_distributed_block(
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    local_gpu_idx,
    target_gnids,
    src_gnids,
    src_part_ids,
    dst_gnids,
    gpu_ranks=None):
    ret = _CAPI_DGLOmegaToDistributedBlock(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        local_gpu_idx,
        F.zerocopy_to_dgl_ndarray(target_gnids),
        F.zerocopy_to_dgl_ndarray(src_gnids),
        F.zerocopy_to_dgl_ndarray(src_part_ids),
        F.zerocopy_to_dgl_ndarray(dst_gnids))
    
    num_assigned_target_nodes = get_num_assigned_targets_per_gpu(
        num_machines, num_gpus_per_machine, target_gnids.shape[0])

    g_idx = ret[0]
    src_gnids_in_block = F.from_dgl_nd(ret[1])
    gpu_rank_in_group = num_gpus_per_machine * machine_rank + local_gpu_idx

    if gpu_ranks is None:
        gpu_ranks = [i for i in range(num_machines * num_gpus_per_machine)]

    dist_block = DGLDistributedBlock(
        gpu_rank_in_group,
        gpu_ranks,
        num_assigned_target_nodes,
        gidx=g_idx,
        ntypes=(['_N'], ['_N']),
        etypes=['_E'])

    assert dist_block.is_unibipartite
    dist_block.srcdata[dgl.NID] = src_gnids_in_block

    return dist_block

def to_block(u, v, dst_ids, src_ids=None):
    if src_ids == None:
        src_ids = empty_tensor

    graph_idx, src_orig_ids = _CAPI_DGLOmegaToBlock(empty_graph._graph,
                                                    F.zerocopy_to_dgl_ndarray(u),
                                                    F.zerocopy_to_dgl_ndarray(v),
                                                    F.zerocopy_to_dgl_ndarray(dst_ids),
                                                    F.zerocopy_to_dgl_ndarray(src_ids))
    block = DGLBlock(graph_idx, (["_N"], ["_N"]), ["_E"])
    block.srcdata[dgl.NID] = F.zerocopy_from_dgl_ndarray(src_orig_ids)
    return block

def partition_request(
    num_machines,
    num_gpus_per_machine,
    nid_partitions,
    target_gnids,
    target_features,
    src_gnids,
    dst_gnids):

    num_targets = target_gnids.shape[0]
    num_tot_gpus = num_machines * num_gpus_per_machine
    num_assigned_targets_per_gpu = get_num_assigned_targets_per_gpu(
        num_machines, num_gpus_per_machine, num_targets)
    target_features_list = target_features.split(num_assigned_targets_per_gpu)

    num_assigned_targets_per_gpu = torch.tensor(num_assigned_targets_per_gpu, dtype=torch.int64)

    ret = _CAPI_DGLOmegaPartitionRequest(
        num_machines,
        num_gpus_per_machine,
        F.zerocopy_to_dgl_ndarray(nid_partitions),
        F.zerocopy_to_dgl_ndarray(num_assigned_targets_per_gpu),
        F.zerocopy_to_dgl_ndarray(target_gnids),
        F.zerocopy_to_dgl_ndarray(src_gnids),
        F.zerocopy_to_dgl_ndarray(dst_gnids))

    src_gnids_list = []
    dst_gnids_list = []

    for i in range(num_tot_gpus):
        src_gnids_list.append(F.from_dgl_nd(ret[i]))
        dst_gnids_list.append(F.from_dgl_nd(ret[num_tot_gpus + i]))

    return (
        [target_gnids] * num_tot_gpus,
        target_features_list,
        src_gnids_list,
        dst_gnids_list
    )

def sample_edges(target_gnids, src_gnids, dst_gnids, fanouts):
    ret = _CAPI_DGLOmegaSampleEdges(
        F.zerocopy_to_dgl_ndarray(target_gnids),
        F.zerocopy_to_dgl_ndarray(src_gnids),
        F.zerocopy_to_dgl_ndarray(dst_gnids),
        F.zerocopy_to_dgl_ndarray(torch.tensor(fanouts, dtype=torch.int64)))

    return [
        (F.from_dgl_nd(ret[2 * i]), F.from_dgl_nd(ret[2 * i + 1])) for i in range(len(fanouts))
    ]

def trace_gen_helper(
    first_new_gnid,
    infer_target_mask,
    batch_local_ids,
    u,
    v,
    u_in_partitions,
    v_in_partitions,
    independent):
    ret = _CAPI_DGLOmegaTraceGenHelper(
        first_new_gnid,
        F.zerocopy_to_dgl_ndarray(infer_target_mask),
        F.zerocopy_to_dgl_ndarray(batch_local_ids),
        F.zerocopy_to_dgl_ndarray(u),
        F.zerocopy_to_dgl_ndarray(v),
        F.zerocopy_to_dgl_ndarray(u_in_partitions),
        F.zerocopy_to_dgl_ndarray(v_in_partitions),
        independent)

    target_gnids = F.from_dgl_nd(ret[0])
    src_gnids = F.from_dgl_nd(ret[1])
    dst_gnids = F.from_dgl_nd(ret[2])

    return target_gnids, src_gnids, dst_gnids

def collect_cpp_traces():
    return _CAPI_DGLOmegaGetCppTraces()

def partition_facebook_dataset(
    num_parts,
    input_dir,
    edge_file_paths,
    include_out_edges,
    infer_prob,
    num_omp_threads
):
    _CAPI_DGLOmegaPartitionFacebookDataset(
        num_parts,
        input_dir,
        edge_file_paths,
        include_out_edges,
        infer_prob,
        num_omp_threads
    )


_init_api("dgl.omega.omega_apis")
