from .envs import ParallelizationType

class GraphServerProcess:

    def __init__(self,
                 channel,
                 num_nodes,
                 node_rank,
                 num_devices_per_node,
                 local_rank,
                 ip_config_path,
                 graph_config_path,
                 parallel_type,
                 using_precomputed_aggregations,
                 precom_filename):
        self._channel = channel
        self._num_nodes = num_nodes
        self._node_rank = node_rank
        self._num_devices_per_node = num_devices_per_node
        self._local_rank = local_rank
        self._ip_config_path = ip_config_path
        self._graph_config_path = graph_config_path

        if parallel_type == ParallelizationType.DATA:
            self._num_clients = num_devices_per_node * 2 * num_nodes
        elif parallel_type == ParallelizationType.DATA or parallel_type == ParallelizationType.P3:
            self._num_clients = num_devices_per_node * 2 * num_nodes
        else:
            self._num_clients = num_devices_per_node * num_nodes + num_nodes
        self._using_precomputed_aggregations = using_precomputed_aggregations
        self._precom_filename = precom_filename
        self._num_servers = 1 # Number of servers for one machin including backup servers
        self._net_type = "socket"
        self._group_id = 0
        self._formats = ["csc"]
        self._keep_alive = False # Whether to keep server alive when clients exit

    def run(self):
        # From dgl.distributed.initialize
        from ..distributed import rpc
        from ..distributed.dist_graph import DistGraphServer

        rpc.reset()

        serv = DistGraphServer(self._node_rank,
                               self._ip_config_path,
                               self._num_servers,
                               self._num_clients,
                               self._graph_config_path,
                               graph_format=self._formats,
                               keep_alive=self._keep_alive,
                               net_type=self._net_type,
                               using_precomputed_aggregations=self._using_precomputed_aggregations,
                               precom_filename=self._precom_filename)

        serv.start(start_callback=lambda: self._channel.notify_initialized())
