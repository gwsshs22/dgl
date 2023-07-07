import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import dgl
from dgl.omega.omega_apis import (
    to_distributed_blocks,
    get_num_assigned_targets_per_gpu)
from dgl import function as fn

from test_utils import create_test_data

def process_main(
    in_queue,
    out_queue,
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    gpu_rank):

    global_rank = machine_rank * num_gpus_per_machine + gpu_rank
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.01:33132',
        rank=global_rank,
        world_size=num_machines * num_gpus_per_machine)
    device = torch.device(f"cuda:{gpu_rank}")
    raw_features, target_gnids, src_gnids, src_part_ids, dst_gnids = in_queue.get()

    dist_block = to_distributed_blocks(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        target_gnids,
        src_gnids,
        src_part_ids,
        dst_gnids)[gpu_rank]
    input_features = raw_features[dist_block.srcdata[dgl.NID]]

    dist_block = dist_block.to(device)
    input_features = input_features.to(device)

    with torch.no_grad():
        while True:
            dist_fn = in_queue.get()
            if dist_fn is None:
                break

            output = dist_fn(dist_block, input_features)
            out_queue.put(output.cpu())

def orig_mean(block, input_features):
    with block.local_scope():
        block.srcdata['h'] = input_features
        block.update_all(fn.copy_u('h', 'm'), fn.mean(msg='m', out='h'))
        return block.dstdata['h']

def dist_mean(dist_block, input_features):
    with dist_block.local_scope():
        dist_block.srcdata['h'] = input_features
        def local_aggr_fn(local_g):
            local_g.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
            return {
                'h': local_g.dstdata['h']
            }

        def merge_fn(aggrs):
            return {
                'h':  aggrs['h'].sum(dim=0)
            }

        dist_block.distributed_message_passing(local_aggr_fn, merge_fn)
        return dist_block.dstdata['h'] / dist_block.in_degrees().reshape(-1, 1)

def test():
    mp.set_start_method('spawn')

    num_machines = 1
    num_gpus_per_machine = 4
    child_processes = []
    in_queues = []
    out_queues = []
    for machine_rank in range(num_machines):
        for gpu_rank in range(num_gpus_per_machine):
            in_queue = mp.Queue()
            out_queue = mp.Queue()
            in_queues.append(in_queue)
            out_queues.append(out_queue)
            p = mp.Process(
                target=process_main,
                args=(
                    in_queue,
                    out_queue,
                    num_machines,
                    machine_rank,
                    num_gpus_per_machine,
                    gpu_rank))
            p.start()
            child_processes.append(p)

    target_gnids, src_gnids, src_part_ids, dst_gnids = create_test_data(
        num_existing_nodes=1000,
        num_target_nodes=50,
        num_machines=num_machines,
        num_connecting_edges=5000,
        random_seed=4132)

    g = dgl.graph((src_gnids, dst_gnids))
    num_nodes = g.number_of_nodes()
    block = dgl.to_block(g, target_gnids)

    raw_features = torch.rand(num_nodes, 128)
    for in_queue in in_queues:
        in_queue.put((raw_features, target_gnids, src_gnids, src_part_ids, dst_gnids))

    input_features = raw_features[block.srcdata[dgl.NID]]
    def test_function(orig_fn, dist_fn=None):
        if dist_fn is None:
            dist_fn = orig_fn

        for in_queue in in_queues:
            in_queue.put(dist_fn)
        
        expected = orig_fn(block, input_features)
        outputs = []
        for out_queue in out_queues:
            outputs.append(out_queue.get())
        output = torch.concat(outputs)
        torch.allclose(expected, output)

    with torch.no_grad():
        test_function(orig_mean, dist_mean)

    for in_queue in in_queues:
        in_queue.put(None)

    for p in child_processes:
        p.join()

if __name__ == "__main__":
    test()
