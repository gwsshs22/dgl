import dgl
import torch # Can implement with NDArrays, but we stick to pytorch now
import torch.distributed as dist
import numpy as np

from .envs import ParallelizationType
from .api import *

class GnnExecutorProcess:
    COMPUTE_REQUEST_TYPE = 0
    P3_OWNER_COMPUTE_REQUEST_TYPE = 1
    P3_OTHER_COMPUTE_REQUEST_TYPE = 2

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
                 using_precomputed_aggregations,
                 graph_name,
                 graph_config_path,
                 model,
                 num_features):
        self._channel = channel
        self._num_nodes = num_nodes
        self._node_rank = node_rank
        self._num_devices_per_node = num_devices_per_node
        self._local_rank = local_rank
        self._master_host = master_host
        self._master_torch_port = master_torch_port
        self._ip_config_path = ip_config_path
        self._parallel_type = parallel_type
        self._load_p3_feature = parallel_type==ParallelizationType.P3
        self._using_precomputed_aggregations = using_precomputed_aggregations
        self._graph_name = graph_name
        self._graph_config_path = graph_config_path
        
        self._device = torch.device(f"cuda:{local_rank}")
        self._cpu_device = torch.device("cpu")
        self._model = model.to(self._device)
        self._num_features = num_features
        self._model.eval()

        self._num_devices_per_node = num_devices_per_node
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
        init_kvstore(self._ip_config_path, self._num_servers, 'default', load_p3_feature=self._load_p3_feature)

        self._dist_graph = dgl.distributed.DistGraph(self._graph_name, part_config=self._graph_config_path)

        if self._load_p3_feature:
            self._dist_graph.load_p3_features(self._num_devices_per_node)
            self._p3_features = self._dist_graph.get_p3_features(self._local_rank)
            self._p3_start_idx, self._p3_end_idx = self._get_p3_start_end_indices()
            print(f"local_rank={self._local_rank}, p3_features shape={self._p3_features.shape}, p3_start_idx={self._p3_start_idx}, p3_end_idx={self._p3_end_idx}")
            self._model.layers[0].p3_split(self._p3_start_idx, self._p3_end_idx)

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
            elif request_type == GnnExecutorProcess.P3_OWNER_COMPUTE_REQUEST_TYPE:
                with torch.no_grad():
                    self.p3_owner_compute(req.batch_id, req.param0)
            elif request_type == GnnExecutorProcess.P3_OTHER_COMPUTE_REQUEST_TYPE:
                with torch.no_grad():
                    self.p3_other_compute(req.batch_id, req.param0)
            else:
                print(f"Unknown request_type={request_type}")
                req.done()
                exit(-1)
            req.done()

    def compute(self, req):
        with torch.no_grad():
            batch_id = req.batch_id
            if self._parallel_type == ParallelizationType.DATA:
                self.data_parallel_compute(batch_id)
            elif self._parallel_type == ParallelizationType.VERTEX_CUT:
                if self._using_precomputed_aggregations:
                    self.vertex_cut_compute_with_precomputed_aggrs(batch_id)
                else:
                    self.vertex_cut_compute(batch_id)


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
        h = self._model.layers[0](block1, h)
        h = self._model.layers[1](block2, h)
        result = h.to("cpu")
        put_tensor(batch_id, "result", result)

    def p3_owner_compute(self, batch_id, owner_gpu_global_rank):
        print(f"p3_owner_compute: self._gpu_global_rank={self._gpu_global_rank}, owner_gpu_global_rank={owner_gpu_global_rank}")
        assert(self._gpu_global_rank == owner_gpu_global_rank)

        new_features = load_tensor(batch_id, "new_features")
        new_features = new_features[:, self._p3_start_idx:self._p3_end_idx]

        input_gnids = load_tensor(batch_id, "input_gnids")

        b1_u = load_tensor(batch_id, "b1_u")
        b1_v = load_tensor(batch_id, "b1_v")
        b2_u = load_tensor(batch_id, "b2_u")
        b2_v = load_tensor(batch_id, "b2_v")

        block1 = dgl.to_block(dgl.graph((b1_u, b1_v))).to(self._device)
        block2 = dgl.to_block(dgl.graph((b2_u, b2_v))).to(self._device)
        
        org_features = self._p3_features[input_gnids[new_features.shape[0]:]]
        h = torch.concat((new_features, org_features)).to(self._device)
        mp_aggr = self._model.layers[0].p3_first_layer_mp(block1, h)
        for k, v in mp_aggr.items():
            dist.reduce(v, owner_gpu_global_rank)
        h = self._model.layers[0].p3_first_layer_dp(block1, mp_aggr)
        h = self._model.layers[1](block2, h)

        result = h.to("cpu")
        put_tensor(batch_id, "result", result)

    def p3_other_compute(self, batch_id, owner_gpu_global_rank):
        new_features = load_tensor(batch_id, "new_features")
        new_features = new_features[:, self._p3_start_idx:self._p3_end_idx]

        input_gnids = load_tensor(batch_id, "input_gnids")
        b1_u = load_tensor(batch_id, "b1_u")
        b1_v = load_tensor(batch_id, "b1_v")

        block1 = dgl.to_block(dgl.graph((b1_u, b1_v))).to(self._device)

        org_features = self._p3_features[input_gnids[new_features.shape[0]:]]
        h = torch.concat((new_features, org_features)).to(self._device)
        mp_aggr = self._model.layers[0].p3_first_layer_mp(block1, h)
        for k, v in mp_aggr.items():
            dist.reduce(v, owner_gpu_global_rank)

    def _get_p3_start_end_indices(self):
        feature_split_in_machines = [self._num_features // self._num_nodes] * self._num_nodes
        for j in range(self._num_features % self._num_nodes):
            feature_split_in_machines[j] += 1
        
        num_features_in_machine = feature_split_in_machines[self._node_rank]
        feature_split_in_gpus = [num_features_in_machine // self._num_devices_per_node] * self._num_devices_per_node
        for j in range(num_features_in_machine % self._num_devices_per_node):
            feature_split_in_gpus[j] += 1 
        
        local_start_idx = 0
        local_end_idx = feature_split_in_gpus[0]
        for j in range(self._local_rank):
            local_start_idx = local_end_idx
            local_end_idx = local_start_idx + feature_split_in_gpus[j+1]
        
        global_start_idx = local_start_idx
        global_end_idx = local_end_idx

        for j in range(self._node_rank):
            global_start_idx += feature_split_in_machines[j]
            global_end_idx += feature_split_in_machines[j]

        return global_start_idx, global_end_idx

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

        if len(input_dst_init_values_map) != 0:
            for k, input_dst_init_values in input_dst_init_values_map.items():
                output_dst_init_val, broadcast_req_handles = self.all_gather_dst_init_values(input_dst_init_values, dst_split)
                for r in broadcast_req_handles:
                    req_handles.append(r)
                output_dst_init_values_map[k] = output_dst_init_val

        for handle in req_handles:
            handle.wait()
        req_handles = []
        # Compute Aggregation
        input_aggs_map = layer.compute_aggregations(block, src_feats, output_dst_init_values_map)
        output_aggs_map = {}

        # All-to-all Aggregation
        for k, input_aggr in input_aggs_map.items():
            output_aggr, req_handle = self.all_to_all_aggregation(input_aggr, dst_split)
            req_handles.append(req_handle)
            output_aggs_map[k] = output_aggr

        for handle in req_handles:
            handle.wait()

        # # Merge all
        return layer.merge(block, src_feats[:num_local_dst_nodes], output_aggs_map)

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

    def vertex_cut_compute_with_precomputed_aggrs(self, batch_id):
        # TODO OPTIMIZATION: can optimize to overlap gpu mem copy & build blocks
        new_gnids = load_tensor(batch_id, "new_gnids")
        new_features = load_tensor(batch_id, "new_features").to(self._device)
        batch_size = new_gnids.shape[0]
        batch_split = self._get_batch_split_to_gpus(batch_size)
        batch_split_cumsum = self._cumsum_start_with_zero(batch_split)

        num_src_nodes_list = load_tensor(batch_id, f"g{self._local_rank}_num_src_nodes_list")

        b2_input_gnids = load_tensor(batch_id, f"g{self._local_rank}_b2_input_gnids")
        dst_split_2 = load_tensor(batch_id, f"dst_split_2")
        b2_u = load_tensor(batch_id, f"g{self._local_rank}_b2_u")
        b2_v = load_tensor(batch_id, f"g{self._local_rank}_b2_v")

        block2 = dgl.to_block(dgl.graph((b2_u, b2_v)), torch.arange(batch_size), src_nodes=torch.arange(num_src_nodes_list[0]), include_dst_in_src=False).to(self._device)

        inc_u = load_tensor(batch_id, f"g{self._local_rank}_inc_u")
        inc_v = load_tensor(batch_id, f"g{self._local_rank}_inc_v")

        inc_dst_gnids = b2_input_gnids[dst_split_2[self._gpu_global_rank]:]
        inc_block = dgl.to_block(dgl.graph((inc_u, inc_v)), torch.arange(inc_dst_gnids.shape[0]), src_nodes=torch.arange(batch_size), include_dst_in_src=False).to(self._device)

        # Build source features
        src_new_feats = new_features[batch_split_cumsum[self._gpu_global_rank]:batch_split_cumsum[self._gpu_global_rank + 1]]
        existing_gnids = b2_input_gnids[batch_split[self._gpu_global_rank]:]
        src_existing_feats = self._dist_graph.ndata["features"][existing_gnids].to(self._device)

        src_feats = torch.concat((src_new_feats, src_existing_feats))
        assert(src_feats.shape[0] == block2.num_src_nodes())

        new_nodes_h = self.vertex_cut_compute_layer(self._model.layers[0], block2, src_feats, dst_split_2)
        inc_computed_h = self.inc_compute(self._model.layers[0], batch_id, inc_block, inc_dst_gnids, new_features)

        h = torch.concat((new_nodes_h, inc_computed_h))
        h = self.vertex_cut_compute_layer(self._model.layers[1], block2, h, dst_split_2)
        put_tensor(batch_id, f"g{self._local_rank}_result", h.to("cpu"))

    def inc_compute(self, first_layer, batch_id, inc_block, inc_dst_gnids, new_features):
        # TODO OPTIMIZATION: precompute some dst value computation for SAGE to remove below dst_features
        dst_features = self._dist_graph.ndata['features'][inc_dst_gnids].to(self._device)
        dst_init_values = {}
        for div_name in first_layer.div_names():
            dst_init_values[div_name] = self._dist_graph.ndata[f"div_{div_name}"][inc_dst_gnids].to(self._device)

        new_nodes_aggregations = first_layer.compute_aggregations(inc_block, new_features, dst_init_values)

        aggrs = {}
        for aggr_name in first_layer.aggr_names():
            aggrs[aggr_name] = torch.stack((new_nodes_aggregations[aggr_name], self._dist_graph.ndata[f"agg_{aggr_name}"][inc_dst_gnids].to(self._device)))

        return first_layer.merge(inc_block, dst_features, aggrs)
