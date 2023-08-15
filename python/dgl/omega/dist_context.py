import torch
import torch.distributed as dist

from .omega_apis import get_num_assigned_targets_per_gpu
from .distributed_block import set_nccl_group

def init_omega(
    master_host,
    master_port,
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    gpu_rank):

    num_workers = num_machines * num_gpus_per_machine
    rank = machine_rank * num_gpus_per_machine + gpu_rank

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{master_host}:{master_port}",
        rank=rank + 1,
        world_size=num_workers + 1)

    nccl_group = dist.new_group(
        ranks=list(range(1, num_workers + 1)),
        backend="nccl")

    set_nccl_group(nccl_group)
    dist.barrier()

def receive_request(num_machines, machine_rank, num_gpus_per_machine, gpu_rank):
    dims = torch.zeros(3, dtype=torch.int64)
    dist.broadcast(dims, 0)

    req_handles = []
    new_gnids = torch.empty((dims[0],), dtype=torch.int64)
    req_handles.append(dist.broadcast(new_gnids, 0, async_op=True))
    src_gnids = torch.empty((dims[2],), dtype=torch.int64)
    req_handles.append(dist.broadcast(src_gnids, 0, async_op=True))
    dst_gnids = torch.empty((dims[2],), dtype=torch.int64)
    req_handles.append(dist.broadcast(dst_gnids, 0, async_op=True))

    global_gpu_rank = machine_rank * num_gpus_per_machine + gpu_rank
    num_targets = get_num_assigned_targets_per_gpu(num_machines, num_gpus_per_machine, dims[0])[global_gpu_rank]
    new_features = torch.empty((num_targets, dims[1]))
    req_handles.append(dist.irecv(new_features, 0))

    for r in req_handles:
        r.wait()

    

    return new_gnids, src_gnids, dst_gnids, new_features

