import dgl
import torch

import time

from .._ffi.function import _init_api
from .. import backend as F

def load_tensor(batch_id, name):
    return F.zerocopy_from_dgl_ndarray(_CAPI_DGLInferenceLoadTensor(batch_id, name))

def put_tensor(batch_id, name, tensor):
    _CAPI_DGLInferencePutTensor(batch_id, name, F.zerocopy_to_dgl_ndarray(tensor))

def split_blocks(block, src_part_ids, dst_part_ids, num_nodes, num_devices_per_node, node_rank, batch_size):
    dst_gnids = block.dstdata[dgl.NID]
    if dst_gnids.shape[0] == batch_size: # last block
        sorted_dst_bids = torch.arange(batch_size)
        sorted_dst_org_ids = block.dstdata[dgl.NID]
    else:
        sorted_dst_bids, sorted_dst_org_ids = sort_dst_ids(num_nodes, num_devices_per_node, batch_size, dst_gnids, dst_part_ids)

    ret = extract_src_ids(num_nodes, num_devices_per_node, node_rank, batch_size, block.srcdata[dgl.NID], src_part_ids)

    blocks = []
    for i in range(num_devices_per_node):
        src_bids = ret[2 * i]
        src_org_ids = ret[2 * i + 1]

        # Slow here. Need to improve
        new_block = dgl.to_block(dgl.graph(block.out_edges(src_bids, 'uv')), sorted_dst_bids, src_nodes=src_bids, include_dst_in_src=False)
        new_block.dstdata[dgl.NID] = sorted_dst_org_ids
        new_block.srcdata[dgl.NID] = src_org_ids
        blocks.append(new_block)
    return blocks

def sort_dst_ids(num_nodes, num_devices_per_node, batch_size, dst_gnids, dst_part_ids):
    part_id_counts = torch.bincount(dst_part_ids)
    ret_list = _CAPI_DGLInferenceSortDstIds(num_nodes,
                                            num_devices_per_node,
                                            batch_size,
                                            F.zerocopy_to_dgl_ndarray(dst_gnids),
                                            F.zerocopy_to_dgl_ndarray(dst_part_ids),
                                            F.zerocopy_to_dgl_ndarray(part_id_counts))
    return F.zerocopy_from_dgl_ndarray(ret_list[0]), F.zerocopy_from_dgl_ndarray(ret_list[1])

def extract_src_ids(num_nodes, num_devices_per_node, node_rank, batch_size, src_gnids, src_part_ids):
    part_id_counts = torch.bincount(src_part_ids)
    ret_list = _CAPI_DGLInferenceExtractSrcIds(num_nodes,
                                               num_devices_per_node,
                                               node_rank,
                                               batch_size,
                                               F.zerocopy_to_dgl_ndarray(src_gnids),
                                               F.zerocopy_to_dgl_ndarray(src_part_ids),
                                               F.zerocopy_to_dgl_ndarray(part_id_counts))
    ret = []
    for r in ret_list:
        ret.append(F.zerocopy_from_dgl_ndarray(r))
    return ret

# _CAPI_DGLInferenceSortDstIds
_init_api("dgl.inference.api")
