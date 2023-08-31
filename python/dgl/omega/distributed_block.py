import sys

import torch
import torch.distributed as dist

from .. import backend as F
from .. import core
from .. import frame
from .. import function as fn
from ..base import ALL, is_all
from ..heterograph import DGLBlock

COMM_ON_HOST = False
NCCL_GROUP = None
GLOO_GROUP = None

def _set_nccl_group(nccl_group):
    global NCCL_GROUP
    NCCL_GROUP = nccl_group

    assert NCCL_GROUP is not None

def _enable_comm_on_host(gloo_group):
    global COMM_ON_HOST
    global GLOO_GROUP

    COMM_ON_HOST = True
    GLOO_GROUP = gloo_group

    assert GLOO_GROUP is not None

class DGLDistributedBlock(DGLBlock):

    def __init__(self,
                 gpu_rank_in_group,
                 gpu_ranks,
                 num_assigned_target_nodes,
                 gidx=[],
                 ntypes=['_N'],
                 etypes=['_E'],
                 node_frames=None,
                 edge_frames=None,
                 **deprecate_kwargs):
        super(DGLDistributedBlock, self).__init__(
            gidx=gidx,
            ntypes=ntypes,
            etypes=etypes,
            node_frames=node_frames,
            edge_frames=edge_frames,
            **deprecate_kwargs)

        self._set_dist_values(gpu_rank_in_group, gpu_ranks, num_assigned_target_nodes)
        assert self._graph.number_of_etypes() == 1, "Distirbuted message passing currently supports homonegeous graphs only"

        etid = 0
        srctype, etype, dsttype = self._canonical_etypes[etid]
        self._stid = self.get_ntype_id_from_src(srctype)
        self._etid = self.get_etype_id((srctype, etype, dsttype))
        self._dtid = self.get_ntype_id_from_dst(dsttype)

        self._node_frames[self._dtid] = frame.Frame(num_rows=self._num_local_target_nodes)

    def _set_dist_values(self, gpu_rank_in_group, gpu_ranks, num_assigned_target_nodes):
        self._gpu_rank_in_group = gpu_rank_in_group
        self._gpu_ranks = gpu_ranks
        self._num_gpus = len(num_assigned_target_nodes)
        self._num_assigned_target_nodes = num_assigned_target_nodes
        self._num_assigned_target_nodes_cumsum = [0] * (self._num_gpus + 1)
        for i in range(self._num_gpus):
            self._num_assigned_target_nodes_cumsum[i + 1] = self._num_assigned_target_nodes_cumsum[i] + \
                self._num_assigned_target_nodes[i]
        self._in_degrees = None
        self._target_start_idx = self._num_assigned_target_nodes_cumsum[self._gpu_rank_in_group]
        self._target_end_idx = self._num_assigned_target_nodes_cumsum[self._gpu_rank_in_group + 1]
        self._num_target_nodes = self._num_assigned_target_nodes_cumsum[-1]
        self._num_local_target_nodes = num_assigned_target_nodes[gpu_rank_in_group]

    @property
    def gpu_rank_in_group(self):
        return self._gpu_rank_in_group

    @property
    def num_assigned_target_nodes(self):
        return self._num_assigned_target_nodes

    def to(self, device, **kwargs):
        if device is None or self.device == device:
            return self

        moved = super(DGLDistributedBlock, self).to(device, **kwargs)
        if moved._in_degrees:
            moved._in_degrees = moved._in_degrees.to(device)
        return moved

    def _assert_context(self):
        assert F.device_type(self.device) != 'cpu', "Distirbuted message passing currently supports GPUs only"

    def in_degrees(self):
        if self._in_degrees is None:
            local_in_degrees = torch.clone(super(DGLDistributedBlock, self).in_degrees())
            dist.all_reduce(local_in_degrees, group=NCCL_GROUP)
            self._in_degrees = local_in_degrees[self._target_start_idx:self._target_end_idx]

        return self._in_degrees
    
    def set_in_degrees(self, global_in_degrees):
        self._in_degrees = global_in_degrees[self._target_start_idx:self._target_end_idx]

    def num_dst_nodes(self, ntype=None):
        assert ntype is None, "Distributed message passing currently supports homogeneous graphs only."
        return self._num_local_target_nodes

    def _set_n_repr(self, ntid, u, data):
        if ntid != self._dtid:
            super(DGLDistributedBlock, self)._set_n_repr(ntid, u, data)
            return

        assert is_all(u), f"Only {ALL} is supported currently. Got {u}"
        self._node_frames[ntid].update(data)

    def _create_local_graph(self):
        etid = 0
        srctype, etype, dsttype = self._canonical_etypes[etid]
        new_g = self._graph.get_relation_graph(etid)

        new_ntypes = ([srctype], [dsttype])
        new_nframes = [
            self._node_frames[self._stid],
            DestinationDataFrameAdapter(
                self,
                self._node_frames[self._dtid],
                self._num_target_nodes)]
        new_etypes = [etype]
        new_eframes = [self._edge_frames[etid]]

        return DGLBlock(new_g, new_ntypes, new_etypes, new_nframes, new_eframes)

    def apply_edges(self, func, edges=ALL, etype=None):
        assert self._graph.number_of_etypes() == 1 or etype is not None, (
            "Distributed message passing currently supports homogeneous graphs only."
        )
        assert is_all(edges), (
            f"Distributed message passing currently supports ALL edges only. Got {edges}"
        )

        eid = ALL
        etid = self.get_etype_id(etype)
        etype = self.canonical_etypes[etid]
        local_g = self._create_local_graph()

        if core.is_builtin(func):
            edata = core.invoke_gsddmm(local_g, func)
        else:
            edata = core.invoke_edge_udf(local_g, eid, etype, func)

        self._set_e_repr(etid, eid, edata)

    def softmax_aggregation(self, logits, msg_field, prob_field, out_field, attn_fn=None):

        def local_aggr_fn(local_g):
            local_g.edata["logits"] = logits
            local_g.update_all(fn.copy_e("logits", "h"), fn.max("h", "logits_max"))

            local_g.apply_edges(fn.e_sub_v("logits", "logits_max", "logits_subtracted"))
            local_g.edata["exp_logits"] = torch.exp(local_g.edata.pop("logits_subtracted"))
            local_g.update_all(fn.copy_e("exp_logits", "h"),
                               fn.sum("h", "exp_logits_sum"))

            # compute softmax
            local_g.apply_edges(fn.e_div_v("exp_logits", "exp_logits_sum", "a"))

            if attn_fn:
                a = local_g.edata.pop("a")
                a = attn_fn(a)
                local_g.edata["a"] = a

            # message passing
            local_g.update_all(fn.u_mul_e(msg_field, 'a', 'm'),
                               fn.sum('m', out_field))

            return {
                "logits_max": local_g.dstdata["logits_max"],
                "exp_logits_sum": local_g.dstdata["exp_logits_sum"],
                out_field: local_g.dstdata[out_field]
            }

        def merge_fn(aggrs):
            logits_max_aggs = aggrs["logits_max"]
            exp_logits_sum_aggs = aggrs["exp_logits_sum"]
            rst_aggs = aggrs[out_field]

            logits_max_aggs[exp_logits_sum_aggs == 0] = -float('inf')
            logits_max, _ = logits_max_aggs.max(0)

            exp_max_logits_diff = torch.exp(logits_max_aggs - logits_max)
            exp_logits_sum_aggs = exp_logits_sum_aggs * exp_max_logits_diff

            rst = rst_aggs * exp_logits_sum_aggs
            rst = rst.sum(0) / exp_logits_sum_aggs.sum(0)
            return {
                out_field: rst
            }

        self.distributed_message_passing(local_aggr_fn, merge_fn)

    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        assert core.is_builtin(message_func) and core.is_builtin(reduce_func), (
            "Distributed message passing currently supports built-in functions only. " +
            "Call distributed_message_passing directly."
        )
        assert apply_node_func == None, "Distributed message passing does not support apply_node_func currently."
        assert etype == None, "Distributed message passing currently supports homogeneous graphs only."

        def local_aggr_fn(local_g):
            local_message_func = message_func
            local_reduce_fun = reduce_func
            if reduce_func.name == "mean":
                local_reduce_fun = fn.sum(msg=reduce_func.msg_field, out=reduce_func.out_field)
            local_g.update_all(local_message_func, local_reduce_fun)
            return {
                "local_aggr": local_g.dstdata[reduce_func.out_field]
            }

        def merge_fn(aggrs):
            if reduce_func.name == "mean" or reduce_func.name == "sum":
                rst = aggrs["local_aggr"].sum(dim=0)
            elif reduce_func.name == "max":
                rst = aggrs["local_aggr"].max(dim=0).values
            elif reduce_func.name == "min":
                rst = aggrs["local_aggr"].min(dim=0).values
            else:
                raise NotImplementedError(f"Unknown reduce function: {reduce_func.name}")
            return {
                reduce_func.out_field: rst
            }

        self.distributed_message_passing(local_aggr_fn, merge_fn)
        if reduce_func.name == "mean":
            rst = self.dstdata[reduce_func.out_field]
            in_degrees = self.in_degrees().reshape(-1, *([1] * (rst.dim()-1)))
            self.dstdata[reduce_func.out_field] = rst / in_degrees

    def distributed_message_passing(
        self,
        local_aggr_fn,
        merge_fn):
        global COMM_ON_HOST

        self._assert_context()
        local_g = self._create_local_graph()
        with local_g.local_scope():
            local_aggrs = local_aggr_fn(local_g)

        req_handles = []
        aggrs = {}
        for k, input_tensor in local_aggrs.items():
            output_tensor, req_handle = self.all_to_all_aggrs(input_tensor)
            aggrs[k] = output_tensor
            req_handles.append(req_handle)

        if COMM_ON_HOST:
            for r in req_handles:
                for t in r:
                    t.wait()
            aggrs = { k: v.to(self.device) for k, v in aggrs.items() }
            
        else:
            for r in req_handles:
                r.wait()

        self._set_n_repr(self._dtid, ALL, merge_fn(aggrs))

    def all_to_all_aggrs(self, input_tensor):
        global COMM_ON_HOST

        if COMM_ON_HOST:
            comm_device = "cpu"
            input_tensor = input_tensor.to("cpu")
        else:
            comm_device = self.device

        output_tensor_dim = (self._num_gpus, self._num_local_target_nodes,)
        for i in range(1, input_tensor.dim()):
            output_tensor_dim += (input_tensor.shape[i],)
        output_tensor = torch.empty(output_tensor_dim, dtype=input_tensor.dtype, device=comm_device)

        inputs = list(input_tensor.split(self._num_assigned_target_nodes))
        outputs = list(output_tensor.split([1] * self._num_gpus))

        if COMM_ON_HOST:
            return output_tensor, self._all_to_all_aggrs_gloo(inputs, outputs)
        else:
            req_handle = dist.all_to_all(outputs, inputs, async_op=True, group=NCCL_GROUP)
            return output_tensor, req_handle

    def _all_to_all_aggrs_gloo(self, inputs, outputs):
        global GLOO_GROUP
        gpu_rank_in_group = self._gpu_rank_in_group
        num_gpus = self._num_gpus

        futs = []
        for i in range(num_gpus):
            if i != gpu_rank_in_group:
                fut = dist.isend(inputs[i], self._gpu_ranks[i], tag=gpu_rank_in_group + i * num_gpus, group=GLOO_GROUP)
                futs.append(fut)

                fut = dist.irecv(outputs[i], self._gpu_ranks[i], tag=i + gpu_rank_in_group * num_gpus, group=GLOO_GROUP)
                futs.append(fut)
        
        outputs[gpu_rank_in_group].copy_(inputs[gpu_rank_in_group])
            
        return futs

    def all_gather_dst_values(self, input_tensor):
        global COMM_ON_HOST
        comm_group = GLOO_GROUP if COMM_ON_HOST else NCCL_GROUP

        if COMM_ON_HOST:
            orig_device = input_tensor.device
            comm_device = "cpu"
            input_tensor = input_tensor.to("cpu")
        else:
            comm_device = self.device

        req_handles = []

        output_tensor_dim = (self._num_target_nodes,)
        for i in range(1, input_tensor.dim()):
            output_tensor_dim += (input_tensor.shape[i],)
        output_tensor = torch.empty(output_tensor_dim, dtype=input_tensor.dtype, device=comm_device)
        outputs = list(output_tensor.split(self._num_assigned_target_nodes))

        for i in range(self._num_gpus):
            if i == self._gpu_rank_in_group:
                req_handles.append(dist.broadcast(input_tensor, self._gpu_ranks[i], async_op=True, group=comm_group))
            else:
                req_handles.append(dist.broadcast(outputs[i], self._gpu_ranks[i], async_op=True, group=comm_group))

        outputs[self._gpu_rank_in_group].copy_(input_tensor)

        for r in req_handles:
            r.wait()
        
        if COMM_ON_HOST:
            output_tensor = output_tensor.to(orig_device)

        return output_tensor

class DestinationDataFrameAdapter(frame.Frame):

    def __init__(self,
                 dist_block,
                 dist_block_dst_frame,
                 num_target_nodes):
        super(DestinationDataFrameAdapter, self).__init__(
            num_rows = num_target_nodes)
        self._dist_block = dist_block
        self._dist_block_dst_frame = dist_block_dst_frame

    def __getitem__(self, name):
        if name not in self._columns:
            local_dst_value = self._dist_block_dst_frame[name]
            dst_value = self._dist_block.all_gather_dst_values(local_dst_value)
            self[name] = dst_value
            return dst_value
        else:
            return super(DestinationDataFrameAdapter, self).__getitem__(name)
