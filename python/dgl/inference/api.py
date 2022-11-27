import dgl
import torch
import numpy as np

import time

from .._ffi.function import _init_api
from .. import backend as F
from ..utils import measure
from ..heterograph import DGLBlock

def load_tensor(batch_id, name):
    return F.zerocopy_from_dgl_ndarray(_CAPI_DGLInferenceLoadTensor(batch_id, name))

def put_tensor(batch_id, name, tensor):
    _CAPI_DGLInferencePutTensor(batch_id, name, F.zerocopy_to_dgl_ndarray(tensor))

def split_local_edges(global_src, global_dst, global_src_part_ids, num_nodes):
    ret_list = _CAPI_DGLInferenceSplitLocalEdges(num_nodes,
                                                F.zerocopy_to_dgl_ndarray(global_src),
                                                F.zerocopy_to_dgl_ndarray(global_dst),
                                                F.zerocopy_to_dgl_ndarray(global_src_part_ids))

    global_src_list = []
    global_dst_list = []

    for i in range(num_nodes):
        global_src_list.append(F.zerocopy_from_dgl_ndarray(ret_list[i]))

    for i in range(num_nodes):
        global_dst_list.append(F.zerocopy_from_dgl_ndarray(ret_list[i + num_nodes]))

    return global_src_list, global_dst_list

def split_blocks(block, src_part_ids, dst_part_ids, num_nodes, num_devices_per_node, node_rank, batch_size):
    dst_gnids = block.dstdata[dgl.NID]
    if dst_gnids.shape[0] == batch_size: # last block
        sorted_dst_bids = torch.arange(batch_size)
        sorted_dst_org_ids = block.dstdata[dgl.NID]
        dst_split = torch.tensor(get_batch_split_to_gpus(batch_size, num_nodes, num_devices_per_node))
    else:
        sorted_dst_bids, sorted_dst_org_ids, dst_split = sort_dst_ids(num_nodes, num_devices_per_node, batch_size, dst_gnids, dst_part_ids)

    subgraphs, src_org_ids_list = _split_blocks(block, num_nodes, num_devices_per_node, node_rank, batch_size, block.srcdata[dgl.NID], src_part_ids, sorted_dst_bids)
    blocks = []
    for i in range(num_devices_per_node):
        blocks.append(DGLBlock(subgraphs[i], (["_N"], ["_N"]), ["_E"]))
        blocks[i].dstdata[dgl.NID] = sorted_dst_org_ids
        blocks[i].srcdata[dgl.NID] = src_org_ids_list[i]

    return blocks, dst_split

def get_batch_split_to_gpus(batch_size, num_nodes, num_devices_per_node):
    num_assigned_to_machines  = np.zeros(num_nodes) + (batch_size // num_nodes)
    for i in range(num_nodes):
        if i < batch_size % num_nodes:
            num_assigned_to_machines[i] += 1
        else:
            break

    split = []
    for machine_idx in range(num_nodes):
        num_assigned_to_machine = num_assigned_to_machines[machine_idx]
        for gpu_idx in range(num_devices_per_node):
            if gpu_idx < num_assigned_to_machine % num_devices_per_node:
                split.append(num_assigned_to_machine // num_devices_per_node + 1)
            else:
                split.append(num_assigned_to_machine // num_devices_per_node)

    return np.array(split, np.int64)

def sort_dst_ids(num_nodes, num_devices_per_node, batch_size, dst_gnids, dst_part_ids):
    part_id_counts = torch.bincount(dst_part_ids)
    ret_list = _CAPI_DGLInferenceSortDstIds(num_nodes,
                                            num_devices_per_node,
                                            batch_size,
                                            F.zerocopy_to_dgl_ndarray(dst_gnids),
                                            F.zerocopy_to_dgl_ndarray(dst_part_ids),
                                            F.zerocopy_to_dgl_ndarray(part_id_counts))
    return F.zerocopy_from_dgl_ndarray(ret_list[0]), F.zerocopy_from_dgl_ndarray(ret_list[1]), F.zerocopy_from_dgl_ndarray(ret_list[2])

def _split_blocks(block, num_nodes, num_devices_per_node, node_rank, batch_size, src_gnids, src_part_ids, sorted_dst_bids):
    part_id_counts = torch.bincount(src_part_ids)
    ret_list = _CAPI_DGLInferenceSplitBlocks(block._graph,
                                             num_nodes,
                                             num_devices_per_node,
                                             node_rank,
                                             batch_size,
                                             F.zerocopy_to_dgl_ndarray(src_gnids),
                                             F.zerocopy_to_dgl_ndarray(src_part_ids),
                                             F.zerocopy_to_dgl_ndarray(part_id_counts),
                                             F.zerocopy_to_dgl_ndarray(sorted_dst_bids))

    sorted_orig_ids_list = []
    for i in range(num_devices_per_node):
        sorted_orig_ids_list.append(F.zerocopy_from_dgl_ndarray(ret_list[i + num_devices_per_node]))
    return ret_list[:num_devices_per_node], sorted_orig_ids_list

# _CAPI_DGLInferenceSortDstIds
_init_api("dgl.inference.api")
