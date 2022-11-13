import dgl
import torch # Can implement with NDArrays, but we stick to pytorch now
import torch.distributed as dist

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

        with torch.no_grad():
            result = self._model([block1, block2], h).to("cpu")
        put_tensor(batch_id, "result", result)

    def vertex_cut_compute(self, batch_id):
        num_dst_nodes_list = load_tensor(batch_id, "num_dst_nodes_list")

        num_src_nodes_list = load_tensor(batch_id, f"g{self._local_rank}_num_src_nodes_list")
        input_gnids = load_tensor(batch_id, f"g{self._local_rank}_input_gnids")
        b1_u = load_tensor(batch_id, f"g{self._local_rank}_b1_u")
        b1_v = load_tensor(batch_id, f"g{self._local_rank}_b1_v")
        b2_u = load_tensor(batch_id, f"g{self._local_rank}_b2_u")
        b2_v = load_tensor(batch_id, f"g{self._local_rank}_b2_v")

        block1 = dgl.to_block(dgl.graph((b1_u, b1_v)), torch.arange(num_dst_nodes_list[0]), src_nodes=torch.arange(num_src_nodes_list[0]), include_dst_in_src=False).to(self._device)
        block2 = dgl.to_block(dgl.graph((b2_u, b2_v)), torch.arange(num_dst_nodes_list[1]), src_nodes=torch.arange(num_src_nodes_list[1]), include_dst_in_src=False).to(self._device)
