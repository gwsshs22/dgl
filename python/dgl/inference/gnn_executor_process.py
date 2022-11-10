import torch # Can implement with NDArrays, but we stick to pytorch now

from .envs import ParallelizationType
from .api import *

class GnnExecutorProcess:
    COMPUTE_REQUEST_TYPE = 0

    def __init__(self, channel, num_nodes, ip_config_path, parallel_type, local_rank):
        self._channel = channel
        self._num_nodes = num_nodes
        self._num_servers = 1 # Number of servers for one machin including backup servers
        self._parallel_type = parallel_type
        self._ip_config_path = ip_config_path
        self._net_type = "socket"
        self._group_id = 0
        self._local_rank = local_rank
        self._device = torch.device(f"cuda:{local_rank}")

    def run(self):
        # From dgl.distributed.initialize

        from ..distributed.constants import MAX_QUEUE_SIZE
        from ..distributed.kvstore import init_kvstore, close_kvstore
        from ..distributed.rpc_client import connect_to_server
        from ..distributed.role import init_role

        connect_to_server(self._ip_config_path, self._num_servers, MAX_QUEUE_SIZE, self._net_type, group_id=self._group_id)
        init_role('default')
        init_kvstore(self._ip_config_path, self._num_servers, 'default')

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
        if self._parallel_type == ParallelizationType.DATA:
            self.data_parallel_compute(req)
        elif self._parallel_type == ParallelizationType.P3:
            req.done()
        elif self._parallel_type == ParallelizationType.VERTEX_CUT:
            req.done()

    def data_parallel_compute(self, req):
        batch_id = req.batch_id
        test_shared_tensor_1_cpu = load_tensor(batch_id, "test_shared_tensor_1_cpu")
        test_shared_tensor_1_gpu = test_shared_tensor_1_cpu.to(self._device)
        final_result = test_shared_tensor_1_gpu * 2

        result = torch.rand(10, 10)
        put_tensor(batch_id, "result", result)
        req.done()
