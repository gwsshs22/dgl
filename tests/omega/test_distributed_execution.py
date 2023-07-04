import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dgl.omega.omega_apis import (
    to_distributed_blocks,
    get_num_assigned_targets_per_gpu)
from dgl import function as fn

from test_utils import create_test_data

def run(
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

    target_gnids, src_gnids, src_part_ids, dst_gnids = create_test_data(
        num_existing_nodes=1000,
        num_target_nodes=50,
        num_machines=num_machines,
        num_connecting_edges=5000,
        random_seed=4132)

    device = torch.device(f"cuda:{gpu_rank}")

    dist_block = to_distributed_blocks(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        target_gnids,
        src_gnids,
        src_part_ids,
        dst_gnids)[gpu_rank].to(device)

    num_assigned_targets_per_gpu = dist_block.num_assigned_target_nodes

    input_dims = 50
    input_feats = torch.ones(dist_block.number_of_src_nodes(), input_dims)
    with dist_block.local_scope():
        # dist_block
        dist_block.srcdata['h'] = input_feats.to(device)

        def local_aggr_fn(local_g):
            local_g.update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'))
            return {
                'h': local_g.dstdata['h'],
                'cnt': local_g.in_degrees()
            }

        def merge_fn(aggrs):
            return {
                'h':  aggrs['h'].sum(dim=0) / aggrs['cnt'].sum(dim=0).reshape(-1, 1)
            }

        dist_block.distributed_message_passing(local_aggr_fn, merge_fn)
        output = dist_block.dstdata['h']

    assert torch.allclose(output, torch.ones_like(output))

def test():
    mp.set_start_method('spawn')

    num_machines = 1
    num_gpus_per_machine = 4
    child_processes = []
    for machine_rank in range(num_machines):
        for gpu_rank in range(num_gpus_per_machine):
            p = mp.Process(
                target=run,
                args=(num_machines, machine_rank, num_gpus_per_machine, gpu_rank))
            p.start()
            child_processes.append(p)

    for p in child_processes:
        p.join()

if __name__ == "__main__":
    test()
