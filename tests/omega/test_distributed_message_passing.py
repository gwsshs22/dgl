import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

import dgl
from dgl.omega.omega_apis import (
    to_distributed_block,
    get_num_assigned_targets_per_gpu)
from dgl.omega.dist_context import set_nccl_group
from dgl import function as fn
import dgl.nn as dglnn

from test_utils import create_test_data

def process_main(
    test_port,
    in_queue,
    out_queue,
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    gpu_rank):

    global_rank = machine_rank * num_gpus_per_machine + gpu_rank
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{test_port}',
        rank=global_rank,
        world_size=num_machines * num_gpus_per_machine)
    set_nccl_group(dist.new_group())
    dist.barrier()
    device = torch.device(f"cuda:{gpu_rank}")
    raw_features, target_gnids, src_gnids, src_part_ids, dst_gnids = in_queue.get()

    dist_block = to_distributed_block(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        gpu_rank,
        target_gnids,
        src_gnids,
        src_part_ids,
        dst_gnids)

    input_features = raw_features[dist_block.srcdata[dgl.NID]]
    dist_block = dist_block.to(device)
    input_features = input_features.to(device)

    with torch.no_grad():
        while True:
            dist_fn = in_queue.get()
            if dist_fn is None:
                break

            output = dist_fn(dist_block, input_features, device).cpu()
            torch.cuda.synchronize(device=device)
            out_queue.put(output)

def identity(block, input_features, device):
    return input_features[:block.number_of_dst_nodes()]

def orig_mean(block, input_features, device):
    with block.local_scope():
        block.srcdata['h'] = input_features
        block.update_all(fn.copy_u('h', 'm'), fn.mean(msg='m', out='h'))
        return block.dstdata['h']

def dist_mean(dist_block, input_features, device):
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

def orig_sum(block, input_features, device):
    with block.local_scope():
        block.srcdata['h'] = input_features
        block.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
        return block.dstdata['h']

def orig_sum_twice(block, input_features, device):
    with block.local_scope():
        block.srcdata['h'] = input_features
        block.srcdata['h2'] = input_features * 2
        block.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
        block.update_all(fn.copy_u('h2', 'm'), fn.sum(msg='m', out='h2'))
        return block.dstdata['h'] + block.dstdata['h2']

def orig_min(block, input_features, device):
    with block.local_scope():
        block.srcdata['h'] = input_features
        block.update_all(fn.copy_u('h', 'm'), fn.min(msg='m', out='h'))
        return block.dstdata['h']

def orig_max(block, input_features, device):
    with block.local_scope():
        block.srcdata['h'] = input_features
        block.update_all(fn.copy_u('h', 'm'), fn.max(msg='m', out='h'))
        return block.dstdata['h']

def orig_gcn(block, input_features, device):
    dim_inputs = input_features.shape[-1]
    dim_hiddens = dim_inputs // 2

    torch.manual_seed(5555)
    gcn_layer = dglnn.GraphConv(dim_inputs, dim_hiddens, allow_zero_in_degree=True)
    gcn_layer = gcn_layer.to(device)

    return gcn_layer(block, input_features)

def orig_sage(block, input_features, device):
    dim_inputs = input_features.shape[-1]
    dim_hiddens = dim_inputs // 2

    torch.manual_seed(5555)
    sage_layer = dglnn.SAGEConv(dim_inputs, dim_hiddens, 'mean')
    sage_layer = sage_layer.to(device)

    return sage_layer(block, input_features)

def orig_gat(block, input_features, device):
    dim_inputs = input_features.shape[-1]
    dim_hiddens = dim_inputs // 2

    torch.manual_seed(5555)
    gat_layer = dglnn.GATConv(
        dim_inputs, dim_hiddens, 4, activation=F.elu, allow_zero_in_degree=True)
    gat_layer = gat_layer.to(device)

    return gat_layer(block, input_features)

def orig_gatv2(block, input_features, device):
    dim_inputs = input_features.shape[-1]
    dim_hiddens = dim_inputs // 2

    torch.manual_seed(5555)
    gat_layer = dglnn.GATv2Conv(
        dim_inputs, dim_hiddens, 4, activation=F.elu, allow_zero_in_degree=True, share_weights=True)
    gat_layer = gat_layer.to(device)

    return gat_layer(block, input_features)

def orig_gatv2_org(block, input_features, device):
    dim_inputs = input_features.shape[-1]
    dim_hiddens = dim_inputs // 2

    torch.manual_seed(5555)
    gat_layer = dglnn.GATv2ConvOrg(
        dim_inputs, dim_hiddens, 4, activation=F.elu, allow_zero_in_degree=True, share_weights=True)
    gat_layer = gat_layer.to(device)

    return gat_layer(block, input_features)

def test(args):
    mp.set_start_method('spawn')

    num_machines = 1
    num_gpus_per_machine = args.num_gpus
    test_port = args.test_port
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
                    test_port,
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
        num_target_nodes=150,
        num_machines=num_machines,
        num_connecting_edges=100000,
        random_seed=4132)
    
    g = dgl.graph((src_gnids, dst_gnids))
    num_nodes = g.number_of_nodes()
    block = dgl.to_block(g, target_gnids)
    raw_features = torch.rand(num_nodes, 256)

    target_gnids.share_memory_()
    src_gnids.share_memory_()
    src_part_ids.share_memory_()
    dst_gnids.share_memory_()
    raw_features.share_memory_()

    for in_queue in in_queues:
        in_queue.put((raw_features, target_gnids, src_gnids, src_part_ids, dst_gnids))

    input_features = raw_features[block.srcdata[dgl.NID]]

    def test_function(orig_fn, dist_fn=None):
        if dist_fn is None:
            dist_fn = orig_fn

        for in_queue in in_queues:
            in_queue.put(dist_fn)

        expected = orig_fn(block, input_features, "cpu")
        outputs = []
        for out_queue in out_queues:
            outputs.append(out_queue.get())
        output = torch.concat(outputs)
        assert torch.allclose(expected, output, atol=1e-5)
    try:
        with torch.no_grad():
            test_function(identity)
            test_function(orig_mean, dist_mean)
            test_function(orig_mean)
            test_function(orig_sum)
            test_function(orig_sum_twice)
            test_function(orig_min)
            test_function(orig_max)
            test_function(orig_gcn)
            test_function(orig_sage)
            test_function(orig_gat)
            test_function(orig_gatv2)
            test_function(orig_gatv2_org, orig_gatv2)
    except AssertionError as e:
        print(f"Test failed: {e}")
        for p in child_processes:
            p.kill()
        return

    print(f"Test passed.")
    for in_queue in in_queues:
        in_queue.put(None)

    for p in child_processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--test_port", type=int, default=34322)

    args = parser.parse_args()
    test(args)
