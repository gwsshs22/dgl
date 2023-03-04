import dgl
import torch # Can implement with NDArrays, but we stick to pytorch now
import torch.distributed as dist
import numpy as np

from .envs import ParallelizationType
from .api import *
from .trace_utils import trace_me, write_traces

class GnnExecutorProcess:
    COMPUTE_REQUEST_TYPE = 0
    P3_OWNER_COMPUTE_REQUEST_TYPE = 1
    P3_OTHER_COMPUTE_REQUEST_TYPE = 2
    WRITE_TRACES = 1000

    def __init__(self,
                 channel,
                 num_nodes,
                 num_backup_servers,
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
                 num_layers,
                 num_features,
                 result_dir,
                 collect_stats):
        self._channel = channel
        self._num_nodes = num_nodes
        self._num_backup_servers = num_backup_servers
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
        self._result_dir = result_dir
        self._collect_stats = collect_stats

        self._device = torch.device(f"cuda:{local_rank}")
        self._cpu_device = torch.device("cpu")
        self._model = model.to(self._device)
        self._num_layers = num_layers
        self._num_features = num_features
        self._model.eval()

        self._num_devices_per_node = num_devices_per_node
        self._num_total_gpus = num_nodes * num_devices_per_node
        self._gpu_global_rank = num_devices_per_node * node_rank + local_rank
        self._num_servers = 1 + num_backup_servers # Number of servers for one machin including backup servers
        self._net_type = "socket"
        self._group_id = 0
        self._num_omp_threads = 32

    def run(self):
        # From dgl.distributed.initialize

        from ..distributed.constants import MAX_QUEUE_SIZE
        from ..distributed.kvstore import init_kvstore, close_kvstore
        from ..distributed.rpc_client import connect_to_server
        from ..distributed.role import init_role
        from ..utils import set_num_threads

        set_num_threads(self._num_omp_threads)
        connect_to_server(self._ip_config_path, self._num_servers, MAX_QUEUE_SIZE, self._net_type, group_id=self._group_id)
        init_role('default')
        is_vcut = (self._parallel_type == ParallelizationType.VERTEX_CUT)
        init_kvstore(self._ip_config_path, self._num_servers, 'default', load_p3_feature=self._load_p3_feature, is_vcut=is_vcut)

        self._dist_graph = dgl.distributed.DistGraph(self._graph_name, part_config=self._graph_config_path)

        if self._load_p3_feature:
            self._dist_graph.load_p3_features(self._num_devices_per_node)
            self._p3_features = self._dist_graph.get_p3_features(self._local_rank)
            self._p3_global_start_idx, self._p3_global_end_idx, \
                self._p3_start_idx, self._p3_end_idx = self._get_start_end_indices(self._num_features)
            self._model.layers[0].p3_split(self._p3_global_start_idx, self._p3_global_end_idx)

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
            elif request_type == GnnExecutorProcess.WRITE_TRACES:
                write_traces(self._result_dir, self._node_rank)
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
        with trace_me(batch_id, "compute"):
            u_list = []
            v_list = []
            blocks = []
            with trace_me(batch_id, "compute/load_tensors"):
                new_features = load_tensor(batch_id, "new_features")
                org_features = load_tensor(batch_id, "org_features")
                for block_idx in range(self._num_layers):
                    u_list.append(load_tensor(batch_id, f"b{block_idx}_u"))
                    v_list.append(load_tensor(batch_id, f"b{block_idx}_v"))

                num_src_nodes_list = load_tensor(batch_id, "num_src_nodes_list").tolist()
                num_dst_nodes_list = load_tensor(batch_id, "num_dst_nodes_list").tolist()

            with trace_me(batch_id, "compute/block_creation"):
                for block_idx in range(self._num_layers):
                    block = dgl.create_block((u_list[block_idx], v_list[block_idx]), num_dst_nodes=num_dst_nodes_list[block_idx], num_src_nodes=num_src_nodes_list[block_idx], check_uv_range=False)
                    blocks.append(block)

            with trace_me(batch_id, "compute/prepare_input"):
                blocks = list(map(lambda b: b.to(self._device), blocks))
                h = torch.concat((new_features, org_features)).to(self._device)
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, "compute/gnn"):
                for block_idx in range(self._num_layers):
                    h = self._model.layers[block_idx](blocks[block_idx], h)
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, "compute/put_tensor"):
                result = h.to("cpu")
                put_tensor(batch_id, "result", result)

    def p3_owner_compute(self, batch_id, owner_gpu_global_rank):
        assert(self._gpu_global_rank == owner_gpu_global_rank)
        u_list = []
        v_list = []
        blocks = []
        with trace_me(batch_id, "compute"):
            with trace_me(batch_id, "compute/load_tensors"):
                new_features = load_tensor(batch_id, "new_features")
                new_features = new_features[:, self._p3_start_idx:self._p3_end_idx]

                input_gnids = load_tensor(batch_id, "input_gnids")

                for block_idx in range(self._num_layers):
                    u_list.append(load_tensor(batch_id, f"b{block_idx}_u"))
                    v_list.append(load_tensor(batch_id, f"b{block_idx}_v"))

                num_src_nodes_list = load_tensor(batch_id, "num_src_nodes_list").tolist()
                num_dst_nodes_list = load_tensor(batch_id, "num_dst_nodes_list").tolist()

            with trace_me(batch_id, "compute/block_creation"):
                for block_idx in range(self._num_layers):
                    block = dgl.create_block((u_list[block_idx], v_list[block_idx]), num_dst_nodes=num_dst_nodes_list[block_idx], num_src_nodes=num_src_nodes_list[block_idx], check_uv_range=False)
                    blocks.append(block)

            with trace_me(batch_id, "compute/prepare_input"):
                blocks = list(map(lambda b: b.to(self._device), blocks))
                org_features = self._p3_features[input_gnids[new_features.shape[0]:]]
                h = torch.concat((new_features, org_features)).to(self._device)
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, "compute/model_parallel"):                
                mp_aggr = self._model.layers[0].p3_first_layer_mp(blocks[0], h)
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, "compute/reduce"):
                handles = []
                for k, v in mp_aggr.items():
                    handles.append(dist.reduce(v, owner_gpu_global_rank, async_op=True))
                for h in handles:
                    h.wait()
                
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, "compute/data_parallel"):
                h = self._model.layers[0].p3_first_layer_dp(blocks[0], mp_aggr)
                for block_idx in range(1, self._num_layers):
                    h = self._model.layers[block_idx](blocks[block_idx], h)
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, "compute/put_tensor"):
                result = h.to("cpu")
                put_tensor(batch_id, "result", result)

    def p3_other_compute(self, batch_id, owner_gpu_global_rank):
        new_features = load_tensor(batch_id, "new_features")
        new_features = new_features[:, self._p3_start_idx:self._p3_end_idx]

        input_gnids = load_tensor(batch_id, "input_gnids")
        b0_u = load_tensor(batch_id, "b0_u")
        b0_v = load_tensor(batch_id, "b0_v")

        num_src_nodes_list = load_tensor(batch_id, "num_src_nodes_list").tolist()
        num_dst_nodes_list = load_tensor(batch_id, "num_dst_nodes_list").tolist()

        block = dgl.create_block((b0_u, b0_v), num_dst_nodes=num_dst_nodes_list[0], num_src_nodes=num_src_nodes_list[0], check_uv_range=False).to(self._device)

        org_features = self._p3_features[input_gnids[new_features.shape[0]:]]
        h = torch.concat((new_features, org_features)).to(self._device)
        mp_aggr = self._model.layers[0].p3_first_layer_mp(block, h)

        for k, v in mp_aggr.items():
            dist.reduce(v, owner_gpu_global_rank)

    def _get_start_end_indices(self, target_size):
        feature_split_in_machines = [target_size // self._num_nodes] * self._num_nodes
        for j in range(target_size % self._num_nodes):
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

        return global_start_idx, global_end_idx, local_start_idx, local_end_idx

    def vertex_cut_compute(self, batch_id):
        with trace_me(batch_id, "compute"):
            with trace_me(batch_id, "compute/load_tensors"):
                # TODO OPTIMIZATION: can optimize to overlap gpu mem copy & compute & computation graph build
                new_gnids = load_tensor(batch_id, "new_gnids")
                new_features = load_tensor(batch_id, "new_features")
                batch_size = new_gnids.shape[0]
                batch_split = self._get_batch_split_to_gpus(batch_size)
                _, _, feat_start_idx, feat_end_idx = self._get_start_end_indices(batch_size)

                num_dst_nodes_list = load_tensor(batch_id, "num_dst_nodes_list").tolist()

                num_src_nodes_list = load_tensor(batch_id, f"g{self._local_rank}_num_src_nodes_list").tolist()
                input_gnids = load_tensor(batch_id, f"g{self._local_rank}_input_gnids")

                dst_split_1 = load_tensor(batch_id, "dst_split_1")
                b1_u = load_tensor(batch_id, f"g{self._local_rank}_b1_u")
                b1_v = load_tensor(batch_id, f"g{self._local_rank}_b1_v")

                dst_split_2 = load_tensor(batch_id, f"dst_split_2")
                b2_u = load_tensor(batch_id, f"g{self._local_rank}_b2_u")
                b2_v = load_tensor(batch_id, f"g{self._local_rank}_b2_v")

            with trace_me(batch_id, "compute/block_creation"):
                block1 = dgl.create_block((b1_u, b1_v), num_dst_nodes=num_dst_nodes_list[0], num_src_nodes=num_src_nodes_list[0], check_uv_range=False)
                block2 = dgl.create_block((b2_u, b2_v), num_dst_nodes=num_dst_nodes_list[1], num_src_nodes=num_src_nodes_list[1], check_uv_range=False)

            with trace_me(batch_id, "compute/prepare_input"):
                block1 = block1.to(self._device)
                block2 = block2.to(self._device)
                # Build source features
                src_new_feats = new_features[feat_start_idx:feat_end_idx].to(self._device)
                existing_gnids = input_gnids[batch_split[self._gpu_global_rank]:]
                src_existing_feats = self._dist_graph.ndata["features"][existing_gnids].to(self._device)

                src_feats = torch.concat((src_new_feats, src_existing_feats))
                assert(src_feats.shape[0] == block1.num_src_nodes())

                h = src_feats
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            h = self.vertex_cut_compute_layer(self._model.layers[0], block1, h, dst_split_1, batch_id, 0)
            h = self.vertex_cut_compute_layer(self._model.layers[1], block2, h, dst_split_2, batch_id, 1)

            with trace_me(batch_id, "compute/put_tensor"):
                put_tensor(batch_id, f"g{self._local_rank}_result", h.to("cpu"))

    def vertex_cut_compute_layer(self, layer, block, src_feats, dst_split, batch_id, layer_idx):
        with trace_me(batch_id, f"compute/layer_{layer_idx}"):
            with trace_me(batch_id, f"compute/layer_{layer_idx}/compute_init_vals"):
                dst_split = dst_split.tolist()
                num_local_dst_nodes = dst_split[self._gpu_global_rank]
                req_handles = []

                # All gather dst init values
                dst_feats = src_feats[:num_local_dst_nodes]
                input_dst_init_values_map = layer.compute_dst_init_values(block, src_feats, num_local_dst_nodes)
                output_dst_init_values_map = {}
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, f"compute/layer_{layer_idx}/comm_init_vals"):
                if len(input_dst_init_values_map) != 0:
                    for k, input_dst_init_values in input_dst_init_values_map.items():
                        output_dst_init_val, broadcast_req_handles = self.all_gather_dst_init_values(input_dst_init_values, dst_split)
                        for r in broadcast_req_handles:
                            req_handles.append(r)
                        output_dst_init_values_map[k] = output_dst_init_val
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, f"compute/layer_{layer_idx}/compute_merge_init_vals"):
                dst_feats = src_feats[:num_local_dst_nodes]
                dst_merge_init_values = layer.compute_dst_merge_init_values(dst_feats)
                for handle in req_handles:
                    handle.wait()
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, f"compute/layer_{layer_idx}/compute_aggregations"):
                req_handles = []
                # Compute Aggregation
                input_aggs_map = layer.compute_aggregations(block, src_feats, output_dst_init_values_map)
                output_aggs_map = {}
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, f"compute/layer_{layer_idx}/comm_aggregations"):
                # All-to-all Aggregation
                for k, input_aggr in input_aggs_map.items():
                    output_aggr, req_handle = self.all_to_all_aggregation(input_aggr, dst_split)
                    req_handles.append(req_handle)
                    output_aggs_map[k] = output_aggr

                for handle in req_handles:
                    handle.wait()

                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            with trace_me(batch_id, f"compute/layer_{layer_idx}/merge"):
                # # Merge all
                ret = layer.merge(block, dst_merge_init_values, output_aggs_map)
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)
                return ret

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

    def _get_vcut_feature_indices(self, batch_size):
        feature_split_in_gpus = [batch_size // self._num_devices_per_node] * self._num_devices_per_node
        for j in range(batch_size % self._num_devices_per_node):
            feature_split_in_gpus[j] += 1

        start_idx = 0
        end_idx = feature_split_in_gpus[0]
        for j in range(self._local_rank):
            start_idx = end_idx
            end_idx = start_idx + feature_split_in_gpus[j+1]

        return start_idx, end_idx

    def vertex_cut_compute_with_precomputed_aggrs(self, batch_id):
        with trace_me(batch_id, "compute"):
            with trace_me(batch_id, "compute/load_tensors"):
                # TODO OPTIMIZATION: can optimize to overlap gpu mem copy & build blocks
                new_gnids = load_tensor(batch_id, "new_gnids")
                new_features = load_tensor(batch_id, "new_features").to(self._device)
                batch_size = new_gnids.shape[0]
                batch_split = self._get_batch_split_to_gpus(batch_size)
                batch_split_cumsum = self._cumsum_start_with_zero(batch_split)

                num_src_nodes_list = load_tensor(batch_id, f"g{self._local_rank}_num_src_nodes_list").tolist()

                b2_input_gnids = load_tensor(batch_id, f"g{self._local_rank}_b2_input_gnids")
                dst_split_2 = load_tensor(batch_id, f"dst_split_2")

                b2_u = load_tensor(batch_id, f"g{self._local_rank}_b2_u")
                b2_v = load_tensor(batch_id, f"g{self._local_rank}_b2_v")

                inc_u = load_tensor(batch_id, f"g{self._local_rank}_inc_u")
                inc_v = load_tensor(batch_id, f"g{self._local_rank}_inc_v")

            with trace_me(batch_id, "compute/block_creation"):
                block2 = dgl.create_block((b2_u, b2_v), num_dst_nodes=batch_size, num_src_nodes=num_src_nodes_list[0], check_uv_range=False)
                inc_dst_gnids = b2_input_gnids[dst_split_2[self._gpu_global_rank]:]

                inc_block = dgl.create_block((inc_u, inc_v), num_dst_nodes=inc_dst_gnids.shape[0], num_src_nodes=batch_size, check_uv_range=False)

            with trace_me(batch_id, "compute/prepare_input"):
                block2 = block2.to(self._device)
                inc_block = inc_block.to(self._device)
                # Build source features
                src_new_feats = new_features[batch_split_cumsum[self._gpu_global_rank]:batch_split_cumsum[self._gpu_global_rank + 1]]
                existing_gnids = b2_input_gnids[batch_split[self._gpu_global_rank]:]
                src_existing_feats = self._dist_graph.ndata["features"][existing_gnids].to(self._device)

                src_feats = torch.concat((src_new_feats, src_existing_feats))
                assert(src_feats.shape[0] == block2.num_src_nodes())
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            new_nodes_h = self.vertex_cut_compute_layer(self._model.layers[0], block2, src_feats, dst_split_2, batch_id, 0)

            with trace_me(batch_id, "compute/inc_compute"):
                inc_computed_h = self.inc_compute(self._model.layers[0], batch_id, inc_block, inc_dst_gnids, new_features)
                h = torch.concat((new_nodes_h, inc_computed_h))
                if self._collect_stats:
                    torch.cuda.synchronize(device=self._device)

            h = self.vertex_cut_compute_layer(self._model.layers[1], block2, h, dst_split_2, batch_id, 1)
            with trace_me(batch_id, "compute/put_tensor"):
                put_tensor(batch_id, f"g{self._local_rank}_result", h.to("cpu"))

    def inc_compute(self, first_layer, batch_id, inc_block, inc_dst_gnids, new_features):
        with trace_me(batch_id, "compute/inc_compute/prepare_input"):
            div_names = first_layer.div_names()
            dmiv_names = first_layer.dmiv_names()
            aggr_names = first_layer.aggr_names()

            dst_init_values = {}
            for div_name in div_names:
                dst_init_values[div_name] = self._dist_graph.ndata[f"div_{div_name}"][inc_dst_gnids].to(self._device)

            dst_merge_init_values = {}
            for dmiv_name in dmiv_names:
                dst_merge_init_values[dmiv_name] = self._dist_graph.ndata[f"dmiv_{dmiv_name}"][inc_dst_gnids].to(self._device)

            precomputed_aggregations = {}
            for aggr_name in aggr_names:
                precomputed_aggregations[aggr_name] = self._dist_graph.ndata[f"agg_{aggr_name}"][inc_dst_gnids].to(self._device)

            if self._collect_stats:
                torch.cuda.synchronize(device=self._device)

        with trace_me(batch_id, "compute/inc_compute/compute_new_nodes_aggregations"):
            new_nodes_aggregations = first_layer.compute_aggregations(inc_block, new_features, dst_init_values)

            aggrs = {}
            for aggr_name in aggr_names:
                aggrs[aggr_name] = torch.stack((new_nodes_aggregations[aggr_name], precomputed_aggregations[aggr_name]))
            if self._collect_stats:
                torch.cuda.synchronize(device=self._device)

        with trace_me(batch_id, "compute/inc_compute/merge"):
            merged_ret = first_layer.merge(inc_block, dst_merge_init_values, aggrs)
            if self._collect_stats:
                torch.cuda.synchronize(device=self._device)
            return merged_ret

    def _cumsum_start_with_zero(self, arr):
            ret = np.zeros(arr.shape[0] + 1, np.int64)
            np.cumsum(arr, out=ret[1:])
            return ret
