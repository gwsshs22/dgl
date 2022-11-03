class GnnExecutorProcess:
    
    def __init__(self, channel, num_nodes, ip_config_path):
        self._channel = channel
        self._num_nodes = num_nodes
        self._num_servers = 1 # Number of servers for one machin including backup servers
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
            self._channel.fetch_request()
