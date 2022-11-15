import torch # Can implement with NDArrays, but we stick to pytorch now
import dgl

from .envs import ParallelizationType
from .api import *

class SamplerProcess:
    SAMPLE_REQUEST_TYPE = 0
    DATA_PARALLEL_INPUT_FETCH = 1

    def __init__(self,
                 channel,
                 num_nodes,
                 node_rank,
                 num_devices_per_node,
                 local_rank,
                 ip_config_path,
                 parallel_type,
                 graph_name,
                 graph_config_path):
        self._channel = channel
        self._num_nodes = num_nodes
        self._node_rank = node_rank
        self._num_devices_per_node = num_devices_per_node
        self._local_rank = local_rank
        self._ip_config_path = ip_config_path
        self._parallel_type = parallel_type
        self._graph_name = graph_name
        self._graph_config_path = graph_config_path

        self._num_servers = 1 # Number of servers for one machin including backup servers
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

        self._dist_graph = dgl.distributed.DistGraph(self._graph_name, part_config=self._graph_config_path)
        self._gpb = self._dist_graph.get_partition_book()

        self._channel.notify_initialized()
        while True:
            req = self._channel.fetch_request()
            request_type = req.request_type
            batch_id = req.batch_id
            if request_type == SamplerProcess.SAMPLE_REQUEST_TYPE:
                if self._parallel_type == ParallelizationType.DATA or self._parallel_type == ParallelizationType.P3:
                    self.sample(batch_id)
                else:
                    self.vcut_sample(batch_id)
            elif request_type == SamplerProcess.DATA_PARALLEL_INPUT_FETCH:
                self.data_parallel_input_fetch(batch_id)
            else:
                print(f"Unknown request_type={request_type}")
                req.done()
                exit(-1)
            req.done()

    def sample(self, batch_id):
        new_gnids = load_tensor(batch_id, "new_gnids")
        src_gnids = load_tensor(batch_id, "src_gnids")
        dst_gnids = load_tensor(batch_id, "dst_gnids")

        batch_size = new_gnids.shape[0]

        input_graph = dgl.graph((src_gnids, dst_gnids))
        second_block_eids = input_graph.in_edges(new_gnids, 'eid')
        second_block = dgl.to_block(input_graph.edge_subgraph(second_block_eids, relabel_nodes=False), new_gnids)

        base_block_eids = input_graph.in_edges(second_block.srcdata[dgl.NID], 'eid')
        base_block = dgl.to_block(input_graph.edge_subgraph(base_block_eids, relabel_nodes=False), second_block.srcdata[dgl.NID])

        first_org_block_seed = second_block.srcdata[dgl.NID][batch_size:]

        frontier = dgl.distributed.sample_neighbors(self._dist_graph, first_org_block_seed, -1)
        first_org_block = dgl.to_block(frontier, first_org_block_seed)

        first_block = self.merge_blocks(base_block, first_org_block, batch_size)

        b1_u, b1_v = first_block.edges()
        b2_u, b2_v = second_block.edges()

        put_tensor(batch_id, "input_gnids", first_block.srcdata[dgl.NID])
        put_tensor(batch_id, "b1_u", b1_u)
        put_tensor(batch_id, "b1_v", b1_v)
        put_tensor(batch_id, "b2_u", b2_u)
        put_tensor(batch_id, "b2_v", b2_v)

    def merge_blocks(self, base_block, org_block, num_shift):
        base_u, base_v = base_block.all_edges()
        org_u, org_v = org_block.all_edges()
        u = torch.concat((base_u, org_u + num_shift))
        v = torch.concat((base_v, org_v + num_shift))

        merged_block = dgl.to_block(dgl.graph((u, v)), torch.arange(num_shift + org_block.num_dst_nodes()))

        merged_block.srcdata[dgl.NID] = torch.concat((base_block.srcdata[dgl.NID][:num_shift], org_block.srcdata[dgl.NID]))
        merged_block.dstdata[dgl.NID] = torch.concat((base_block.dstdata[dgl.NID][:num_shift], org_block.dstdata[dgl.NID]))

        return merged_block

    def vcut_sample(self, batch_id):
        new_gnids = load_tensor(batch_id, "new_gnids")
        src_gnids = load_tensor(batch_id, "src_gnids")
        dst_gnids = load_tensor(batch_id, "dst_gnids")

        batch_size = new_gnids.shape[0]

        input_graph = dgl.graph((src_gnids, dst_gnids))
        second_block_eids = input_graph.in_edges(new_gnids, 'eid')
        second_block = dgl.to_block(input_graph.edge_subgraph(second_block_eids, relabel_nodes=False), new_gnids)

        base_block_eids = input_graph.in_edges(second_block.srcdata[dgl.NID], 'eid')
        base_block = dgl.to_block(input_graph.edge_subgraph(base_block_eids, relabel_nodes=False), second_block.srcdata[dgl.NID])

        first_org_block_seed = second_block.srcdata[dgl.NID][batch_size:]

        frontier = dgl.distributed.sample_neighbors(self._dist_graph, first_org_block_seed, -1)
        first_org_block = dgl.to_block(frontier, first_org_block_seed)

        first_block = self.merge_blocks(base_block, first_org_block, batch_size)

        first_blocks, first_dst_split = split_blocks(first_block,
                                    self._gpb.nid2partid(first_block.srcdata[dgl.NID]),
                                    self._gpb.nid2partid(first_block.dstdata[dgl.NID]),
                                    self._num_nodes,
                                    self._num_devices_per_node,
                                    self._node_rank,
                                    batch_size)

        second_blocks, second_dst_split = split_blocks(second_block,
                                     self._gpb.nid2partid(second_block.srcdata[dgl.NID]),
                                     self._gpb.nid2partid(second_block.dstdata[dgl.NID]),
                                     self._num_nodes,
                                     self._num_devices_per_node,
                                     self._node_rank,
                                     batch_size)

        num_dst_nodes_list = torch.tensor((first_blocks[0].num_dst_nodes(), second_blocks[0].num_dst_nodes()))

        put_tensor(batch_id, "num_dst_nodes_list", num_dst_nodes_list)
        put_tensor(batch_id, "dst_split_1", first_dst_split)
        put_tensor(batch_id, "dst_split_2", second_dst_split)

        for i in range(self._num_devices_per_node):
            fb = first_blocks[i]
            sb = second_blocks[i]
            b1_u, b1_v = fb.edges()
            b2_u, b2_v = sb.edges()

            num_src_nodes_list = torch.tensor((fb.num_src_nodes(), sb.num_src_nodes()))
            put_tensor(batch_id, f"g{i}_num_src_nodes_list", num_src_nodes_list)

            put_tensor(batch_id, f"g{i}_input_gnids", fb.srcdata[dgl.NID])
            put_tensor(batch_id, f"g{i}_b1_u", b1_u)
            put_tensor(batch_id, f"g{i}_b1_v", b1_v)
            put_tensor(batch_id, f"g{i}_b2_u", b2_u)
            put_tensor(batch_id, f"g{i}_b2_v", b2_v)

    def data_parallel_input_fetch(self, batch_id):
        # It will not be called.
        pass
