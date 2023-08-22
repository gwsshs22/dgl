import argparse
from dataclasses import dataclass
import heapq
import os
import sys
import threading
import queue

import torch
import torch.distributed.rpc as rpc
import numpy as np

import dgl
from dgl.omega.dist_context import set_nccl_group
from dgl.omega.omega_apis import (
    get_num_assigned_targets_per_gpu,
    to_block,
    to_distributed_block
)

from models import create_model

@dataclass(frozen=True)
class ModelConfig:
    gnn: str
    num_inputs: int
    num_hiddens: int
    num_classes: int
    num_layers: int
    gat_heads: str
    fanouts: str

class WorkerAsyncExecContext:

    def __init__(
        self,
        master_ip,
        master_dist_comm_port,
        num_machines,
        num_gpus_per_machine,
        part_config,
        graph_name,
        machine_rank,
        global_rank,
        local_rank,
        exec_mode,
        use_precoms,
        model_config,
        random_seed):

        self._req_queue = queue.Queue()
        self._process_thread = threading.Thread(
            target=process_main,
            args=(
                self._req_queue,
                master_ip,
                master_dist_comm_port,
                num_machines,
                num_gpus_per_machine,
                part_config,
                graph_name,
                machine_rank,
                global_rank,
                local_rank,
                exec_mode,
                use_precoms,
                model_config,
                random_seed))
        self._process_thread.start()

    @rpc.functions.async_execution
    def process(self, req_id, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        fut = torch.futures.Future()
        self._req_queue.put((req_id, batch_id, target_gnids, target_features, src_gnids, dst_gnids, fut))
        return fut

def process_main(
    req_queue,
    master_ip,
    master_dist_comm_port,
    num_machines,
    num_gpus_per_machine,
    part_config,
    graph_name,
    machine_rank,
    global_rank,
    local_rank,
    exec_mode,
    use_precoms,
    model_config,
    random_seed):

    pending_requests = {}
    req_id_heap = []
    expected_req_id = 0

    dist_g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
    dist_g.barrier()

    gpu_ranks = None
    if exec_mode == "cgp" or exec_mode == "cgp-multi":
        world_size = num_machines * num_gpus_per_machine
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_ip}:{master_dist_comm_port}",
            world_size=world_size,
            rank=global_rank)

        if exec_mode == "cgp":
            gpu_ranks = [r for r in range(world_size)]
            set_nccl_group(torch.distributed.new_group())
        else:
            for i in range(num_gpus_per_machine):
                ranks_in_group = [r for r in range(i, world_size + i, num_gpus_per_machine)]
                new_group = torch.distributed.new_group(ranks=ranks_in_group)
                if i == local_rank:
                    gpu_ranks = ranks_in_group
                    set_nccl_group(new_group)

        torch.distributed.barrier()

    exec_context = LocalExecutionContext(
        num_machines,
        num_gpus_per_machine,
        machine_rank,
        global_rank,
        local_rank,
        gpu_ranks,
        exec_mode,
        use_precoms,
        model_config,
        random_seed,
        dist_g)

    while True:
        request = req_queue.get()
        req_id = request[0]
        assert req_id >= expected_req_id
        if expected_req_id == req_id:
            expected_req_id += 1
            exec_context.process_request(*request)
        else:
            # Handle out of order arrivals.
            pending_requests[req_id] = request
            heapq.heappush(req_id_heap, req_id)

        while True:
            if req_id_heap and req_id_heap[0] == expected_req_id:
                req_id = heapq.heappop(req_id_heap)
                request = pending_requests[req_id]
                del pending_requests[req_id]
                expected_req_id += 1
                exec_context.process_request(*request)
            else:
                break

class LocalExecutionContext:

    def __init__(
        self,
        num_machines,
        num_gpus_per_machine,
        machine_rank,
        global_rank,
        local_rank,
        gpu_ranks,
        exec_mode,
        use_precoms,
        model_config,
        random_seed,
        dist_g):

        self._num_machines = num_machines
        self._num_gpus_per_machine = num_gpus_per_machine
        self._machine_rank = machine_rank
        self._global_rank = global_rank
        self._local_rank = local_rank
        self._gpu_ranks = gpu_ranks
        self._exec_mode = exec_mode
        self._use_precoms = use_precoms
        self._model_config = model_config
        self._dist_g = dist_g

        self._gpb = self._dist_g.get_partition_book()
        self._device = f"cuda:{self._local_rank}"

        fanouts = [int(fanout) for fanout in model_config.fanouts.split(",")]
        self._fanouts = [-1 if f == 0 else f for f in fanouts]

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self._load_model()
        self._load_precoms()

    def _load_model(self):
        gat_heads = [int(h) for h in self._model_config.gat_heads.split(",")]
        self._model = create_model(
            self._model_config.gnn,
            self._model_config.num_inputs,
            self._model_config.num_hiddens,
            self._model_config.num_classes,
            self._model_config.num_layers,
            gat_heads)

        self._model = self._model.to(self._device)

    def _load_precoms(self):
        if not self._use_precoms:
            return

        self._precoms_dist_tensors = [self._dist_g.ndata["features"]]

        for layer_idx in range(self._model_config.num_layers - 1):
            self._precoms_dist_tensors.append(self._dist_g.ndata[f"layer_{layer_idx}"])

    def process_request(self, req_id, batch_id, target_gnids, target_features, src_gnids, dst_gnids, fut):
        with torch.no_grad():
            if self._exec_mode == "dp":
                if self._use_precoms:
                    ret_tensor = self.run_dp_with_precoms(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
                else:
                    self.run_dp(batch_id, target_gnids, target_features, src_gnids, dst_gnids)
            else:
                ret_tensor = self.run_cgp(batch_id, target_gnids, target_features, src_gnids, dst_gnids)

            fut.set_result((batch_id, ret_tensor))

    def run_dp(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        return torch.rand(target_gnids.shape[0], self._model_config.num_classes)

    def run_dp_with_precoms(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        num_layers = self._model_config.num_layers

        block = to_block(src_gnids, dst_gnids, target_gnids).to(self._device)
        blocks = [block] * num_layers
        inputs = []
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                input_nodes = blocks[layer_idx].srcdata[dgl.NID][block.num_dst_nodes():]
            else:
                block = blocks[layer_idx]
                input_nodes = block.srcdata[dgl.NID][block.num_dst_nodes():]
            inputs.append(self._precoms_dist_tensors[layer_idx][input_nodes])

        inputs = [i.to(self._device) for i in inputs]
        target_features = target_features.to(self._device)
        h = torch.concat((target_features, inputs[0]))
        for layer_idx in range(num_layers):
            h = self._model.layer_foward(layer_idx, blocks[layer_idx], h)
            if layer_idx != num_layers - 1:
                h = torch.concat((h, inputs[layer_idx + 1]))

        return h.cpu()

    def run_cgp(self, batch_id, target_gnids, target_features, src_gnids, dst_gnids):
        # Set num_gpus_per_machine as 1 for cgp-multi.
        num_targets = target_gnids.shape[0]
        is_cgp = self._exec_mode == "cgp" # or cgp-multi
        num_gpus_per_machine_in_group = self._num_gpus_per_machine if is_cgp else 1
        local_rank_in_group = self._local_rank if is_cgp else 0
        num_assigned_targets_per_gpu = get_num_assigned_targets_per_gpu(
            self._num_machines, num_gpus_per_machine_in_group, num_targets)

        rank_in_group = self._global_rank if is_cgp else self._machine_rank
        num_layers = self._model_config.num_layers
        num_local_targets = num_assigned_targets_per_gpu[rank_in_group]
        src_part_ids = self._gpb.nid2partid(src_gnids)

        block = to_distributed_block(
            self._num_machines,
            self._machine_rank,
            num_gpus_per_machine_in_group,
            local_rank_in_group,
            target_gnids,
            src_gnids,
            src_part_ids,
            dst_gnids,
            gpu_ranks=self._gpu_ranks).to(self._device)

        blocks = [block] * num_layers
        inputs = []
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                input_nodes = blocks[layer_idx].srcdata[dgl.NID][block.num_dst_nodes():]
            else:
                block = blocks[layer_idx]
                input_nodes = block.srcdata[dgl.NID][block.num_dst_nodes():]
            inputs.append(self._precoms_dist_tensors[layer_idx][input_nodes])

        inputs = [i.to(self._device) for i in inputs]
        target_features = target_features.to(self._device)
        h = torch.concat((target_features, inputs[0]))
        for layer_idx in range(num_layers):
            h = self._model.layer_foward(layer_idx, blocks[layer_idx], h)
            if layer_idx != num_layers - 1:
                h = torch.concat((h, inputs[layer_idx + 1]))

        return h.cpu()

def main(args):
    num_machines = args.num_machines
    machine_rank = args.machine_rank
    num_gpus_per_machine = args.num_gpus_per_machine
    local_rank = args.local_rank

    world_size = num_machines * num_gpus_per_machine
    global_rank = num_gpus_per_machine * machine_rank + local_rank

    print(f"worker-{global_rank} dgl initializing...", file=sys.stderr)

    dgl.distributed.initialize(
        args.ip_config,
        net_type=args.net_type)

    print(f"worker-{global_rank} dgl initialized.", file=sys.stderr)

    os.environ["MASTER_ADDR"] = str(args.master_ip)
    os.environ["MASTER_PORT"] = str(args.master_rpc_port)

    rpc.init_rpc(f"worker-{global_rank}", rank=global_rank + 1, world_size=world_size + 1)
    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--net_type', type=str, default='socket',
                        help="backend net type, 'socket' or 'tensorpipe'")

    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--master_rpc_port', type=int)
    parser.add_argument('--master_dist_comm_port', type=int)
    parser.add_argument('--num_machines', type=int)
    parser.add_argument('--machine_rank', type=int)
    parser.add_argument('--num_gpus_per_machine', type=int)
    parser.add_argument('--local_rank', type=int)

    args = parser.parse_args()

    main(args)
