import dgl
import torch # Can implement with NDArrays, but we stick to pytorch now
import torch.distributed as dist
import numpy as np

from .envs import ParallelizationType
from .api import *

class GnnExecutorProcess:
    COMPUTE_REQUEST_TYPE = 0

    def __init__(self,
                 channel,
                 num_nodes,
                 node_rank,
                 num_devices_per_node,
                 local_rank,
                 master_host,
                 master_torch_port,
                 ip_config_path,
                 parallel_type,
                 graph_name,
                 graph_config_path,
                 model):
        self._channel = channel
        self._num_nodes = num_nodes
        self._node_rank = node_rank
        self._num_devices_per_node = num_devices_per_node
        self._local_rank = local_rank
        self._master_host = master_host
        self._master_torch_port = master_torch_port
        self._ip_config_path = ip_config_path
        self._parallel_type = parallel_type
        self._graph_name = graph_name
        self._graph_config_path = graph_config_path
        
        self._device = torch.device(f"cuda:{local_rank}")
        self._cpu_device = torch.device("cpu")
        self._model = model.to(self._device)
        self._model.eval()

        self._num_total_gpus = num_nodes * num_devices_per_node
        self._gpu_global_rank = num_devices_per_node * node_rank + local_rank
        self._num_servers = 1 # Number of servers for one machin including backup servers
        self._net_type = "socket"
        self._group_id = 0

    def run(self):
        # From dgl.distributed.initialize

        from ..distributed.constants import MAX_QUEUE_SIZE
        from ..distributed.kvstore import init_kvstore, close_kvstore
        from ..distributed.rpc_client import connect_to_server
        from ..distributed.role import init_role

        connect_to_server(self._ip_config_path, self._num_servers, MAX_QUEUE_SIZE, self._net_type, group_id=self._group_id)
        init_role('default')
        init_kvstore(self._ip_config_path, self._num_servers, 'default')

        self._dist_graph = dgl.distributed.DistGraph(self._graph_name, part_config=self._graph_config_path)

        global_rank = self._node_rank * self._num_devices_per_node + self._local_rank
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{self._master_host}:{self._master_torch_port}',
            rank=global_rank,
            world_size=self._num_nodes * self._num_devices_per_node)
        self._group_gloo = dist.new_group(backend="gloo")

        self._channel.notify_initialized()
        while True:
            req = self._channel.fetch_request()

            request_type = req.request_type
            if request_type == GnnExecutorProcess.COMPUTE_REQUEST_TYPE:
                self.compute(req)
            else:
                print(f"Unknown request_type={request_type}")
                exit(-1)

    def compute(self, req):
        with torch.no_grad():
            batch_id = req.batch_id
            if self._parallel_type == ParallelizationType.DATA:
                self.data_parallel_compute(batch_id)
            elif self._parallel_type == ParallelizationType.P3:
                pass
            elif self._parallel_type == ParallelizationType.VERTEX_CUT:
                self.vertex_cut_compute(batch_id)
            req.done()

    def data_parallel_compute(self, batch_id):
        new_features = load_tensor(batch_id, "new_features")

        input_gnids = load_tensor(batch_id, "input_gnids")
        b1_u = load_tensor(batch_id, "b1_u")
        b1_v = load_tensor(batch_id, "b1_v")
        b2_u = load_tensor(batch_id, "b2_u")
        b2_v = load_tensor(batch_id, "b2_v")
        
        block1 = dgl.to_block(dgl.graph((b1_u, b1_v))).to(self._device)
        block2 = dgl.to_block(dgl.graph((b2_u, b2_v))).to(self._device)

        org_features = self._dist_graph.ndata["features"][input_gnids[new_features.shape[0]:]]
        h = torch.concat((new_features, org_features)).to(self._device)

        result = self._model([block1, block2], h).to("cpu")
        put_tensor(batch_id, "result", result)

    def vertex_cut_compute(self, batch_id):
        # TODO OPTIMIZATION: can optimize to overlap gpu mem copy & build blocks
        new_gnids = load_tensor(batch_id, "new_gnids")
        new_features = load_tensor(batch_id, "new_features")
        batch_size = new_gnids.shape[0]
        batch_split = self._get_batch_split_to_gpus(batch_size)
        batch_split_cumsum = self._cumsum_start_with_zero(batch_split)

        num_dst_nodes_list = load_tensor(batch_id, "num_dst_nodes_list")

        num_src_nodes_list = load_tensor(batch_id, f"g{self._local_rank}_num_src_nodes_list")
        input_gnids = load_tensor(batch_id, f"g{self._local_rank}_input_gnids")

        dst_split_1 = load_tensor(batch_id, "dst_split_1")
        b1_u = load_tensor(batch_id, f"g{self._local_rank}_b1_u")
        b1_v = load_tensor(batch_id, f"g{self._local_rank}_b1_v")

        dst_split_2 = load_tensor(batch_id, f"dst_split_2")
        b2_u = load_tensor(batch_id, f"g{self._local_rank}_b2_u")
        b2_v = load_tensor(batch_id, f"g{self._local_rank}_b2_v")

        block1 = dgl.to_block(dgl.graph((b1_u, b1_v)), torch.arange(num_dst_nodes_list[0]), src_nodes=torch.arange(num_src_nodes_list[0]), include_dst_in_src=False).to(self._device)
        block2 = dgl.to_block(dgl.graph((b2_u, b2_v)), torch.arange(num_dst_nodes_list[1]), src_nodes=torch.arange(num_src_nodes_list[1]), include_dst_in_src=False).to(self._device)

        # Build source features
        src_new_feats = new_features[batch_split_cumsum[self._gpu_global_rank]:batch_split_cumsum[self._gpu_global_rank + 1]].to(self._device)
        existing_gnids = input_gnids[batch_split[self._gpu_global_rank]:]
        src_existing_feats = self._dist_graph.ndata["features"][existing_gnids].to(self._device)

        src_feats = torch.concat((src_new_feats, src_existing_feats))
        assert(src_feats.shape[0] == block1.num_src_nodes())

        h = src_feats
        h = self.vertex_cut_compute_layer(self._model.layers[0], block1, h, dst_split_1)
        h = self.vertex_cut_compute_layer(self._model.layers[1], block2, h, dst_split_2)

        put_tensor(batch_id, f"g{self._local_rank}_result", h.to("cpu"))

    def vertex_cut_compute_layer(self, layer, block, src_feats, dst_split):
        dst_split = dst_split.tolist()
        num_local_dst_nodes = dst_split[self._gpu_global_rank]
        req_handles = []

        # All gather dst init values
        dst_feats = src_feats[:num_local_dst_nodes]
        input_dst_init_values_map = layer.compute_dst_init_values(block, src_feats, num_local_dst_nodes)
        output_dst_init_values_map = {}

        if input_dst_init_values_map is not None:
            for k, input_dst_init_values in input_dst_init_values_map.items():
                output_dst_init_val, broadcast_req_handles = self.all_gather_dst_init_values(input_dst_init_values, dst_split)
                for r in broadcast_req_handles:
                    req_handles.append(r)
                output_dst_init_values_map[k] = output_dst_init_val

        for handle in req_handles:
            handle.wait()
        req_handles = []
        # Compute Aggregation
        input_aggs_map = layer.compute_aggregations(block, src_feats, num_local_dst_nodes, output_dst_init_values_map)
        output_aggs_map = {}

        # All-to-all Aggregation
        for k, input_aggr in input_aggs_map.items():
            output_aggr, req_handle = self.all_to_all_aggregation(input_aggr, dst_split)
            req_handles.append(req_handle)
            output_aggs_map[k] = output_aggr

        for handle in req_handles:
            handle.wait()

        # # Merge all
        return layer.merge(block, src_feats, num_local_dst_nodes, output_aggs_map)

    def all_gather_dst_init_values(self, input_tensor, dst_split):
        # TODO OPTIMIZATION: check whether this multiple calls of broadcast underperform or not.
        # We use multiple broadcasts because all_gather does not support the tensors with different lengths.
        req_handles = []

        output_tensor_dim = (np.sum(dst_split),)
        for i in range(1, input_tensor.dim()):
            output_tensor_dim += (input_tensor.shape[i],)
        output_tensor = torch.zeros(output_tensor_dim, dtype=input_tensor.dtype, device=self._device)
        outputs = list(output_tensor.split(dst_split))

        for gpu_idx in range(self._num_total_gpus):
            if gpu_idx == self._gpu_global_rank:
                req_handles.append(dist.broadcast(input_tensor, gpu_idx, async_op=True))
            else:
                req_handles.append(dist.broadcast(outputs[gpu_idx], gpu_idx, async_op=True))

        outputs[self._gpu_global_rank].copy_(input_tensor) # non_blocking=True does not work for device-device mem copy. Can we make it asynchronous?        

        return output_tensor, req_handles

    def all_to_all_aggregation(self, input_tensor, dst_split):
        output_tensor_dim = (self._num_total_gpus, dst_split[self._gpu_global_rank],)
        for i in range(1, input_tensor.dim()):
            output_tensor_dim += (input_tensor.shape[i],)
        output_tensor = torch.empty(output_tensor_dim, dtype=input_tensor.dtype, device=self._device)

        inputs = list(input_tensor.split(dst_split))
        outputs = list(output_tensor.split([1] * self._num_total_gpus))

        req_handle = dist.all_to_all(outputs, inputs, async_op=True)
        return output_tensor, req_handle

    def _get_batch_split_to_gpus(self, total_count):
        num_assigned_to_machines  = np.zeros(self._num_nodes) + (total_count // self._num_nodes)
        for i in range(self._num_nodes):
            if i < total_count % self._num_nodes:
                num_assigned_to_machines[i] += 1
            else:
                break

        split = []
        for machine_idx in range(self._num_nodes):
            num_assigned_to_machine = num_assigned_to_machines[machine_idx]
            for gpu_idx in range(self._num_devices_per_node):
                if gpu_idx < num_assigned_to_machine % self._num_devices_per_node:
                    split.append(num_assigned_to_machine // self._num_devices_per_node + 1)
                else:
                    split.append(num_assigned_to_machine // self._num_devices_per_node)

        return np.array(split, np.int64)
    
    def _cumsum_start_with_zero(self, arr):
        ret = np.zeros(arr.shape[0] + 1, np.int64)
        np.cumsum(arr, out=ret[1:])
        return ret
