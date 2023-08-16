import torch
import torch.distributed as dist

import dgl
from dgl import backend as F

from .._ffi.function import _init_api
from ..sampling import sample_neighbors as local_sample_neighbors

def dist_sample_neighbors(
    g,
    seed,
    fanout,
    batch_id,
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    gpu_rank):
    # with trace_me(batch_id, "vcut_sample_neighbors"):

        # with trace_me(batch_id, "vcut_sample_neighbors/local_sampling"):
    num_gpus = num_machines * num_gpus_per_machine
    global_gpu_rank = machine_rank * num_gpus_per_machine + gpu_rank
    local_graph = g.local_partition
    gpb = g.get_partition_book()
    part_ids = gpb.nid2partid(seed)

    logcal_gnids_mask = torch.logical_and(
        part_ids == machine_rank,
        seed % num_gpus_per_machine == gpu_rank)
    local_gnids = F.boolean_mask(seed, logcal_gnids_mask)
    local_nids = gpb.nid2localnid(local_gnids, machine_rank)

    if fanout == -1:
        src, dst = local_graph.in_edges(local_nids)
    else:
        sg = local_sample_neighbors(local_graph, local_nids, fanout, _dist_training=True)
        src, dst = sg.edges()

    global_nid_mapping = local_graph.ndata[dgl.NID]
    global_src, global_dst = F.gather_row(global_nid_mapping, src), \
        F.gather_row(global_nid_mapping, dst)

    global_src_part_ids = gpb.nid2partid(global_src)

        # with trace_me(batch_id, "vcut_sample_neighbors/split_local_edges"):
    global_src_list, global_dst_list = _split_local_edges(
        global_src,
        global_dst,
        global_src_part_ids,
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        gpu_rank)

        # with trace_me(batch_id, "vcut_sample_neighbors/all_to_all_edges"):
    num_edges_per_partition = torch.tensor(list(map(lambda l: len(l), global_src_list)), dtype=torch.int64)

    all_num_edges_per_partition = torch.zeros((num_gpus * num_gpus), dtype=torch.int64)
    dist.all_gather(list(all_num_edges_per_partition.split([num_gpus] * num_gpus)), num_edges_per_partition)
    all_num_edges_per_partition = all_num_edges_per_partition.reshape(num_gpus, num_gpus)
    expected_num_per_partition = all_num_edges_per_partition.transpose(0, 1)[global_gpu_rank].tolist()

    global_src_output = []
    global_dst_output = []
    for i in range(num_gpus):
        if i == global_gpu_rank:
            global_src_output.append(global_src_list[global_gpu_rank])
            global_dst_output.append(global_dst_list[global_gpu_rank])
        else:
            global_src_output.append(torch.zeros(expected_num_per_partition[i], dtype=torch.int64))
            global_dst_output.append(torch.zeros(expected_num_per_partition[i], dtype=torch.int64))

    req_handles = []

    req_handles.extend(_all_to_all_edges(global_src_output, global_src_list, num_gpus, global_gpu_rank, batch_id, 0))
    req_handles.extend(_all_to_all_edges(global_dst_output, global_dst_list, num_gpus, global_gpu_rank, batch_id, 1))

    for r in req_handles:
        r.wait()

    ret_list = []
    for i in range(num_gpus):
        ret_list.append((global_src_output[i], global_dst_output[i]))
    return ret_list

def _split_local_edges(
    global_src,
    global_dst,
    global_src_part_ids,
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    gpu_rank):

    ret_list = _CAPI_DGLOmegaSplitLocalEdges(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        gpu_rank,
        F.zerocopy_to_dgl_ndarray(global_src),
        F.zerocopy_to_dgl_ndarray(global_dst),
        F.zerocopy_to_dgl_ndarray(global_src_part_ids))

    global_src_list = []
    global_dst_list = []

    num_gpus = num_machines * num_gpus_per_machine
    for i in range(num_gpus):
        global_src_list.append(F.zerocopy_from_dgl_ndarray(ret_list[i]))

    for i in range(num_gpus):
        global_dst_list.append(F.zerocopy_from_dgl_ndarray(ret_list[i + num_gpus]))

    return global_src_list, global_dst_list

def _all_to_all_edges(outputs, inputs, num_gpus, global_gpu_rank, batch_id, op_tag):
    def make_tag(i, j):
        return (batch_id << 10) + (op_tag << 8) + (i << 4) + j

    req_handle_list = []
    for i in range(num_gpus):
        if i == global_gpu_rank:
            continue
        req_handle_list.append(dist.isend(inputs[i], i, tag=make_tag(global_gpu_rank, i)))
        req_handle_list.append(dist.irecv(outputs[i], i, tag=make_tag(i, global_gpu_rank)))

    return req_handle_list

_init_api("dgl.omega.dist_sample")
