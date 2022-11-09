from .api import *

class SamplerProcess:
    SAMPLE_REQUEST_TYPE = 0
    CLEANUP_REQUEST_TYPE = 1

    def __init__(self, channel, num_nodes, ip_config_path):
        self._channel = channel
        self._num_nodes = num_nodes
        self._num_servers = 1 # Number of servers for one machin including backup servers
        self._ip_config_path = ip_config_path
        self._net_type = "socket"
        self._group_id = 0
        self._num_omp_threads = 1

    def run(self):
        # From dgl.distributed.initialize

        from ..distributed.constants import MAX_QUEUE_SIZE
        from ..distributed.kvstore import init_kvstore, close_kvstore
        from ..distributed.rpc_client import connect_to_server
        from ..distributed.role import init_role
        from ..utils import set_num_threads

        set_num_threads(self._num_omp_threads)
        connect_to_server(self._ip_config_path, self._num_servers, MAX_QUEUE_SIZE, self._net_type, group_id=self._group_id)
        init_role('sampler')
        init_kvstore(self._ip_config_path, self._num_servers, 'sampler')

        self._channel.notify_initialized()
        while True:
            req = self._channel.fetch_request()
            if req.request_type == SamplerProcess.SAMPLE_REQUEST_TYPE:
                self.sample(req)
            else:
                self.cleanup(req)

    def sample(self, req):
        batch_id = req.batch_id
        new_ngids = load_tensor(batch_id, "new_ngids")
        src_ngids = load_tensor(batch_id, "src_ngids")
        dst_ngids = load_tensor(batch_id, "dst_ngids")
        print(f"[{batch_id}] new_ngids={new_ngids}")
        req.done()
    
    def cleanup(self, req):

        req.done()
