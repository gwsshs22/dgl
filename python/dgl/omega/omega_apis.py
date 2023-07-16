import dgl

from .distributed_block import DGLDistributedBlock
from .._ffi.function import _init_api
from .. import backend as F
from .. import utils


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
    for gpu_idx in range(num_gpus_per_machine):
        g_idx = ret[2 * gpu_idx]
        src_gnids_in_block = F.from_dgl_nd(ret[2 * gpu_idx + 1])
        global_gpu_rank = num_gpus_per_machine * machine_rank + gpu_idx

        dist_block = DGLDistributedBlock(
            global_gpu_rank,
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
    dst_gnids):
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
    global_gpu_rank = num_gpus_per_machine * machine_rank + local_gpu_idx

    dist_block = DGLDistributedBlock(
        global_gpu_rank,
        num_assigned_target_nodes,
        gidx=g_idx,
        ntypes=(['_N'], ['_N']),
        etypes=['_E'])

    assert dist_block.is_unibipartite
    dist_block.srcdata[dgl.NID] = src_gnids_in_block

    return dist_block

_init_api("dgl.omega.omega_apis")
