from .envs import ParallelizationType

class GnnExecutorProcess:
    COMPUTE_REQUEST_TYPE = 0
    CLEANUP_REQUEST_TYPE = 99999

    def __init__(self, channel, num_nodes, ip_config_path, parallel_type):
        self._channel = channel
        self._num_nodes = num_nodes
        self._num_servers = 1 # Number of servers for one machin including backup servers
        self._parallel_type = parallel_type
        self._ip_config_path = ip_config_path
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

        self._channel.notify_initialized()
        while True:
            req = self._channel.fetch_request()

            request_type = req.request_type
            if request_type == GnnExecutorProcess.COMPUTE_REQUEST_TYPE:
                self.compute(req)
            elif request_type == GnnExecutorProcess.CLEANUP_REQUEST_TYPE:
                self.cleanup(req)
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

    def cleanup(self, req):
        req.done()

    def data_parallel_compute(self, req):
        batch_id = req.batch_id
        print(f"[batch_id={batch_id}] data_parallel_compute")
        req.done()
