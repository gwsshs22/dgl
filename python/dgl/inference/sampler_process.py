from collections import namedtuple

import torch # Can implement with NDArrays, but we stick to pytorch now
import dgl
import torch.distributed as dist

from .envs import ParallelizationType
from .api import *
from .trace_utils import trace_me, write_traces
from ..utils import measure
from dgl import backend as F

LocalSampledGraph = namedtuple('LocalSampledGraph', 'global_src global_dst')

class SamplerProcess:
    SAMPLE_REQUEST_TYPE = 0
    DATA_PARALLEL_INPUT_FETCH = 1
    WRITE_TRACES = 1000

    def __init__(self,
                 channel,
                 num_nodes,
                 num_backup_servers,
                 node_rank,
                 num_devices_per_node,
                 local_rank,
                 master_host,
                 ip_config_path,
                 parallel_type,
                 using_precomputed_aggregations,
                 graph_name,
                 graph_config_path,
                 num_layers,
                 fanouts,
                 result_dir):
        self._channel = channel
        self._num_nodes = num_nodes
        self._num_backup_servers = num_backup_servers
        self._node_rank = node_rank
        self._num_devices_per_node = num_devices_per_node
        self._local_rank = local_rank
        self._master_host = master_host
        self._gpu_global_rank = num_devices_per_node * node_rank + local_rank
        self._ip_config_path = ip_config_path
        self._parallel_type = parallel_type
        self._using_precomputed_aggregations = using_precomputed_aggregations
        self._graph_name = graph_name
        self._graph_config_path = graph_config_path
        self._num_layers = num_layers
        self._fanouts = fanouts
        self._result_dir = result_dir

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
        init_role('sampler')
        init_kvstore(self._ip_config_path, self._num_servers, 'sampler')

        self._dist_graph = dgl.distributed.DistGraph(self._graph_name, part_config=self._graph_config_path)
        self._gpb = self._dist_graph.get_partition_book()

        if self._parallel_type == ParallelizationType.VERTEX_CUT:
            dist.init_process_group(
                backend='gloo',
                init_method=f'tcp://{self._master_host}:{15000 + self._local_rank}',
                rank=self._node_rank,
                world_size=self._num_nodes)

        self._channel.notify_initialized()
        while True:
            req = self._channel.fetch_request()
            request_type = req.request_type
            batch_id = req.batch_id
            if request_type == SamplerProcess.SAMPLE_REQUEST_TYPE:
                if self._parallel_type == ParallelizationType.DATA or self._parallel_type == ParallelizationType.P3:
                    self.sample(batch_id)
                else:
                    if self._using_precomputed_aggregations:
                        self.vcut_sample_with_precomputed_aggrs(batch_id)
                    else:
                        self.vcut_sample(batch_id)
            elif request_type == SamplerProcess.DATA_PARALLEL_INPUT_FETCH:
                self.data_parallel_input_fetch(batch_id)
            elif request_type == SamplerProcess.WRITE_TRACES:
                write_traces(self._result_dir, self._node_rank)
            else:
                print(f"Unknown request_type={request_type}")
                req.done()
                exit(-1)
            req.done()

    def sample_neighbors(self, seed, fanout):
        return dgl.distributed.sample_neighbors(self._dist_graph, seed, fanout, include_eid=False, return_only_edges=True)

    def merge_edges(self, res_list):
        """Merge request from multiple servers"""
        if len(res_list) > 1:
            srcs = []
            dsts = []
            for res in res_list:
                srcs.append(res.global_src)
                dsts.append(res.global_dst)
            src_tensor = F.cat(srcs, 0)
            dst_tensor = F.cat(dsts, 0)
        else:
            src_tensor = res_list[0].global_src
            dst_tensor = res_list[0].global_dst

        return src_tensor, dst_tensor

    def sample(self, batch_id): 
        blocks = []
        with trace_me(batch_id, "sample"):
            with trace_me(batch_id, "sample/load_tensors"):
                new_gnids = load_tensor(batch_id, "new_gnids")
                src_gnids = load_tensor(batch_id, "src_gnids")
                dst_gnids = load_tensor(batch_id, "dst_gnids")

            with trace_me(batch_id, "sample/blocks"):
                with trace_me(batch_id, f"sample/blocks/create_block_{self._num_layers - 1}"):
                    batch_size = new_gnids.shape[0]

                    # input_graph = dgl.graph((src_gnids, dst_gnids), num_nodes=new_gnids[-1] + 1)
                    # u, v = input_graph.in_edges(new_gnids, 'uv')
                    u, v = fast_in_edges(src_gnids, dst_gnids, new_gnids)
                    last_block = fast_to_block(u, v, new_gnids)
                    blocks.insert(0, last_block)

                    base_src, base_dst = fast_in_edges(src_gnids, dst_gnids, last_block.srcdata[dgl.NID])
                    base_edges = LocalSampledGraph(base_src, base_dst)

                for block_idx in range(self._num_layers - 2, -1, -1):
                    with trace_me(batch_id, f"sample/blocks/create_block_{block_idx}"):
                        with trace_me(batch_id, f"sample/blocks/create_block_{block_idx}/sample_neighbors"):
                            seed = blocks[0].srcdata[dgl.NID][batch_size:]
                            sampled_edges = self.sample_neighbors(seed, self._fanouts[block_idx])

                        with trace_me(batch_id, f"sample/blocks/create_block_{block_idx}/to_block"):
                            # base_src, base_dst = input_graph.in_edges(second_block.srcdata[dgl.NID], 'uv')
                            
                            sampled_edges.insert(0, base_edges)

                            u, v = self.merge_edges(sampled_edges)
                            block = fast_to_block(u, v, blocks[0].srcdata[dgl.NID])
                            blocks.insert(0, block)

            input_gnids = blocks[0].srcdata[dgl.NID]
            if self._parallel_type == ParallelizationType.DATA:
                with trace_me(batch_id, "sample/pull_features"):
                    new_features = load_tensor(batch_id, "new_features")
                    
                    org_features = self._dist_graph.ndata["features"][input_gnids[new_features.shape[0]:]]

            with trace_me(batch_id, "sample/put_tensors"):
                num_src_nodes_list = []
                num_dst_nodes_list = []
                for block_idx, block in enumerate(blocks):
                    u, v = block.edges()
                    put_tensor(batch_id, f"b{block_idx}_u", u)
                    put_tensor(batch_id, f"b{block_idx}_v", v)
                    num_src_nodes_list.append(block.number_of_src_nodes())
                    num_dst_nodes_list.append(block.number_of_dst_nodes())

                put_tensor(batch_id, "num_src_nodes_list", torch.tensor(num_src_nodes_list))
                put_tensor(batch_id, "num_dst_nodes_list", torch.tensor(num_dst_nodes_list))

                if self._parallel_type == ParallelizationType.DATA:
                    put_tensor(batch_id, "org_features", org_features)
                else:
                    put_tensor(batch_id, "input_gnids", input_gnids)

    def vcut_sample(self, batch_id):
        with trace_me(batch_id, "sample"):
            with trace_me(batch_id, "sample/load_tensors"):
                new_gnids = load_tensor(batch_id, "new_gnids")
                src_gnids = load_tensor(batch_id, "src_gnids")
                dst_gnids = load_tensor(batch_id, "dst_gnids")

            with trace_me(batch_id, "sample/blocks"):
                with trace_me(batch_id, "sample/blocks/create_second_block"):
                    batch_size = new_gnids.shape[0]

                    # input_graph = dgl.graph((src_gnids, dst_gnids), num_nodes=new_gnids[-1] + 1)
                    # u, v = input_graph.in_edges(new_gnids, 'uv')
                    u, v = fast_in_edges(src_gnids, dst_gnids, new_gnids)
                    second_block = fast_to_block(u, v, new_gnids)
                    second_block.dstdata[dgl.NID] = new_gnids

                with trace_me(batch_id, "sample/blocks/split_second_block"):
                    second_blocks, second_dst_split = split_blocks(second_block,
                                                self._gpb.nid2partid(second_block.srcdata[dgl.NID]),
                                                self._gpb.nid2partid(second_block.dstdata[dgl.NID]),
                                                self._num_nodes,
                                                self._num_devices_per_node,
                                                self._node_rank,
                                                batch_size)

                with trace_me(batch_id, "sample/blocks/create_first_block"):
                    with trace_me(batch_id, "sample/blocks/create_first_block/sample_neighbors"):
                        first_org_block_seed = second_block.srcdata[dgl.NID][batch_size:]
                        sampled_edges = self.vcut_sample_neighbors(first_org_block_seed, -1, batch_id)
                    with trace_me(batch_id, "sample/blocks/create_first_block/to_block"):
                        base_src, base_dst = fast_in_edges(src_gnids, dst_gnids, second_block.srcdata[dgl.NID])
                        base_edges = LocalSampledGraph(base_src, base_dst)
                        sampled_edges.insert(0, base_edges)

                        u, v = self.merge_edges(sampled_edges)
                        first_block = fast_to_block(u, v, second_block.srcdata[dgl.NID])
                        first_block.dstdata[dgl.NID] = second_block.srcdata[dgl.NID]
                with trace_me(batch_id, "sample/blocks/split_first_block"):
                    first_blocks, first_dst_split = split_blocks(first_block,
                                                self._gpb.nid2partid(first_block.srcdata[dgl.NID]),
                                                self._gpb.nid2partid(first_block.dstdata[dgl.NID]),
                                                self._num_nodes,
                                                self._num_devices_per_node,
                                                self._node_rank,
                                                batch_size)
            with trace_me(batch_id, "sample/put_tensors"):
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

    def vcut_sample_neighbors(self, seed, fanout, batch_id):
        with trace_me(batch_id, "vcut_sample_neighbors"):
            with trace_me(batch_id, "vcut_sample_neighbors/local_sampling"):
                assert fanout == -1, "Currently only support full sampling"
                local_graph = self._dist_graph.local_partition
                part_ids = self._gpb.nid2partid(seed)
                local_gnids = F.boolean_mask(seed, part_ids == self._node_rank)
                local_nids = self._gpb.nid2localnid(local_gnids, self._node_rank)

                global_nid_mapping = local_graph.ndata[dgl.NID]
                src, dst = local_graph.in_edges(local_nids)
                global_src, global_dst = F.gather_row(global_nid_mapping, src), \
                    F.gather_row(global_nid_mapping, dst)

                global_src_part_ids = self._gpb.nid2partid(global_src)

            with trace_me(batch_id, "vcut_sample_neighbors/split_local_edges"):
                global_src_list, global_dst_list = split_local_edges(global_src, global_dst, global_src_part_ids, self._num_nodes)

            with trace_me(batch_id, "vcut_sample_neighbors/all_to_all_edges"):
                num_edges_per_partition = torch.tensor(list(map(lambda l: len(l), global_src_list)), dtype=torch.int64)

                all_num_edges_per_partition = torch.zeros((self._num_nodes * self._num_nodes), dtype=torch.int64)
                dist.all_gather(list(all_num_edges_per_partition.split([self._num_nodes] * self._num_nodes)), num_edges_per_partition)
                all_num_edges_per_partition = all_num_edges_per_partition.reshape(self._num_nodes, self._num_nodes)
                expected_num_per_partition = all_num_edges_per_partition.transpose(0, 1)[self._node_rank].tolist()

                global_src_output = []
                global_dst_output = []
                for i in range(self._num_nodes):
                    if i == self._node_rank:
                        global_src_output.append(global_src_list[self._node_rank])
                        global_dst_output.append(global_dst_list[self._node_rank])
                    else:
                        global_src_output.append(torch.zeros(expected_num_per_partition[i], dtype=torch.int64))
                        global_dst_output.append(torch.zeros(expected_num_per_partition[i], dtype=torch.int64))

                req_handles = []

                req_handles.extend(self.all_to_all_edges(global_src_output, global_src_list, batch_id, 0))
                req_handles.extend(self.all_to_all_edges(global_dst_output, global_dst_list, batch_id, 1))

                for r in req_handles:
                    r.wait()

                ret_list = []
                for i in range(self._num_nodes):
                    ret_list.append(LocalSampledGraph(global_src_output[i], global_dst_output[i]))
                return ret_list

    def all_to_all_edges(self, outputs, inputs, batch_id, op_tag):
        def make_tag(i, j):
            return (batch_id << 10) + (op_tag << 8) + (i << 4) + j

        req_handle_list = []
        for i in range(self._num_nodes):
            if i == self._node_rank:
                continue
            req_handle_list.append(dist.isend(inputs[i], i, tag=make_tag(self._node_rank, i)))
            req_handle_list.append(dist.irecv(outputs[i], i, tag=make_tag(i, self._node_rank)))

        return req_handle_list

    def vcut_sample_with_precomputed_aggrs(self, batch_id):
        with trace_me(batch_id, "sample"):
            with trace_me(batch_id, "sample/load_tensors"):
                new_gnids = load_tensor(batch_id, "new_gnids")
                src_gnids = load_tensor(batch_id, "src_gnids")
                dst_gnids = load_tensor(batch_id, "dst_gnids")

            with trace_me(batch_id, "sample/blocks"):
                with trace_me(batch_id, "sample/blocks/create_second_block"):
                    # input_graph = dgl.graph((src_gnids, dst_gnids), num_nodes=new_gnids[-1] + 1)
                    # u, v = input_graph.in_edges(new_gnids, 'uv')
                    u, v = fast_in_edges(src_gnids, dst_gnids, new_gnids)
                    second_block = fast_to_block(u, v, new_gnids)
                    second_block.dstdata[dgl.NID] = new_gnids

                with trace_me(batch_id, "sample/blocks/split_second_block"):
                    batch_size = new_gnids.shape[0]
                    second_blocks, second_dst_split = split_blocks(second_block,
                                                self._gpb.nid2partid(second_block.srcdata[dgl.NID]),
                                                self._gpb.nid2partid(second_block.dstdata[dgl.NID]),
                                                self._num_nodes,
                                                self._num_devices_per_node,
                                                self._node_rank,
                                                batch_size)

                with trace_me(batch_id, "sample/blocks/create_inc_block"):
                    inc_comp_blocks = []
                    for i in range(self._num_devices_per_node):
                        inc_comp_dst_gnids = second_blocks[i].srcdata[dgl.NID][second_dst_split[i]:]
                        inc_comp_u, inc_comp_v = fast_in_edges(src_gnids, dst_gnids, inc_comp_dst_gnids)
                        inc_comp_block = fast_to_block(inc_comp_u, inc_comp_v, inc_comp_dst_gnids, new_gnids)
                        inc_comp_blocks.append(inc_comp_block)

            with trace_me(batch_id, "sample/put_tensors"):
                put_tensor(batch_id, "dst_split_2", second_dst_split)

                for i in range(self._num_devices_per_node):
                    sb = second_blocks[i]
                    ib = inc_comp_blocks[i]

                    b2_u, b2_v = sb.edges()
                    inc_u, inc_v = ib.edges()

                    num_src_nodes_list = torch.tensor([sb.num_src_nodes()])

                    put_tensor(batch_id, f"g{i}_num_src_nodes_list", num_src_nodes_list)
                    put_tensor(batch_id, f"g{i}_b2_input_gnids", sb.srcdata[dgl.NID])
                    put_tensor(batch_id, f"g{i}_b2_u", b2_u)
                    put_tensor(batch_id, f"g{i}_b2_v", b2_v)

                    put_tensor(batch_id, f"g{i}_inc_u", inc_u)
                    put_tensor(batch_id, f"g{i}_inc_v", inc_v)

    def data_parallel_input_fetch(self, batch_id):
        # It will not be called.
        pass
