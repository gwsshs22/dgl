import torch
import torch.distributed as dist

from ..heterograph import DGLBlock
from .. import backend as F
from .. import frame
from ..base import ALL, is_all

class DGLDistributedBlock(DGLBlock):

    def __init__(self,
                 global_gpu_rank,
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

        self._set_dist_values(global_gpu_rank, num_assigned_target_nodes)
        assert self._graph.number_of_etypes() == 1, "Distirbuted message passing currently supports homonegeous graphs only"

        etid = 0
        srctype, etype, dsttype = self._canonical_etypes[etid]
        self._stid = self.get_ntype_id_from_src(srctype)
        self._etid = self.get_etype_id((srctype, etype, dsttype))
        self._dtid = self.get_ntype_id_from_dst(dsttype)

        self._node_frames[self._dtid] = frame.Frame(num_rows=self._num_local_target_nodes)


    def _set_dist_values(self, global_gpu_rank, num_assigned_target_nodes):
        self._global_gpu_rank = global_gpu_rank
        self._num_gpus = len(num_assigned_target_nodes)
        self._num_assigned_target_nodes = num_assigned_target_nodes
        self._num_assigned_target_nodes_cumsum = [0] * (self._num_gpus + 1)
        for i in range(self._num_gpus):
            self._num_assigned_target_nodes_cumsum[i + 1] = self._num_assigned_target_nodes_cumsum[i] + \
                self._num_assigned_target_nodes[i]
        self._in_degrees = None
        self._target_start_idx = self._num_assigned_target_nodes_cumsum[self._global_gpu_rank]
        self._target_end_idx = self._num_assigned_target_nodes_cumsum[self._global_gpu_rank + 1]
        self._num_target_nodes = self._num_assigned_target_nodes_cumsum[-1]
        self._num_local_target_nodes = num_assigned_target_nodes[global_gpu_rank]

    @property
    def global_gpu_rank(self):
        return self._global_gpu_rank

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
            local_in_degrees = super(DGLDistributedBlock, self).in_degrees()
            dist.all_reduce(local_in_degrees)
            self._in_degrees = local_in_degrees[self._target_start_idx:self._target_end_idx]

        return self._in_degrees

    def _set_n_repr(self, ntid, u, data):
        if ntid != self._dtid:
            super(DGLDistributedBlock, self)._set_n_repr(ntid, u, data)
            return

        assert is_all(u), f"Only {ALL} is supported currently. Got {u}"
        num_nodes = self._num_local_target_nodes
        self._node_frames[ntid].update(data)

    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        raise NotImplementedError("Call distributed_message_passing directly.")

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

    def distributed_message_passing(
        self,
        local_aggr_fn,
        merge_fn):

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

        for r in req_handles:
            r.wait()

        self._set_n_repr(self._dtid, ALL, merge_fn(aggrs))

    def all_to_all_aggrs(self, input_tensor):
        output_tensor_dim = (self._num_gpus, self._num_local_target_nodes,)
        for i in range(1, input_tensor.dim()):
            output_tensor_dim += (input_tensor.shape[i],)
        output_tensor = torch.empty(output_tensor_dim, dtype=input_tensor.dtype, device=self.device)

        inputs = list(input_tensor.split(self._num_assigned_target_nodes))
        outputs = list(output_tensor.split([1] * self._num_gpus))

        req_handle = dist.all_to_all(outputs, inputs, async_op=True)
        return output_tensor, req_handle

    def all_gather_dst_values(self, input_tensor):
        req_handles = []

        output_tensor_dim = (self._num_target_nodes,)
        for i in range(1, input_tensor.dim()):
            output_tensor_dim += (input_tensor.shape[i],)
        output_tensor = torch.zeros(output_tensor_dim, dtype=input_tensor.dtype, device=self._device)
        outputs = list(output_tensor.split(self._num_assigned_target_nodes))

        for gpu_idx in range(self._num_gpus):
            if gpu_idx == self._global_gpu_rank:
                req_handles.append(dist.broadcast(input_tensor, gpu_idx, async_op=True))
            else:
                req_handles.append(dist.broadcast(outputs[gpu_idx], gpu_idx, async_op=True))

        outputs[self._global_gpu_rank].copy_(input_tensor)

        for r in req_handles:
            r.wait()

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
