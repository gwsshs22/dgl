import argparse
import time

from dgl.omega.omega_apis import get_num_assigned_targets_per_gpu

import torch
import torch.distributed as dist

def main(args):
    num_machines = args.num_machines
    num_gpus_per_machine = args.num_gpus_per_machine
    num_workers = args.num_machines * num_gpus_per_machine
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{args.master_host}:{args.master_port}",
        rank=0,
        world_size=num_workers + 1)

    dist.new_group(
        ranks=list(range(1, num_workers + 1)),
        backend="nccl")

    dist.barrier()

    for _ in range(10):
        batch_size = 1024
        feature_dim = 602
        new_gnids = torch.arange(batch_size, dtype=torch.int64)
        src_gnids = torch.arange(batch_size, dtype=torch.int64)
        dst_gnids = torch.arange(batch_size, dtype=torch.int64)
        new_features = torch.rand(batch_size, feature_dim)


        t1 = time.time()
        num_targets_per_gpu = get_num_assigned_targets_per_gpu(num_machines, num_gpus_per_machine, batch_size)
        new_features = new_features.split(num_targets_per_gpu)

        num_edges = new_gnids.shape[0]

        req_handles = []
        req_handles.append(dist.broadcast(
            torch.tensor([batch_size, feature_dim, num_edges], dtype=torch.int64), 0, async_op=True))
        req_handles.append(dist.broadcast(new_gnids, 0, async_op=True))
        req_handles.append(dist.broadcast(src_gnids, 0, async_op=True))
        req_handles.append(dist.broadcast(dst_gnids, 0, async_op=True))

        for i in range(num_workers):
            req_handles.append(dist.isend(new_features[i], i + 1))

        for r in req_handles:
            r.wait()
        
        t2 = time.time()
        print(f"Broadcast = {t2-t1:.4f}s")

        req_handles = []
        output = torch.empty((batch_size, args.num_outputs))
        output_split = output.split(num_targets_per_gpu)
        for i in range(num_workers):
            req_handles.append(dist.irecv(output_split[i], i + 1))
        for r in req_handles:
            r.wait()
        
        t3 = time.time()
        print(f"Fetch = {t3-t2:.4f}s")
        print(f"output={output}, output.shape={output.shape}")


    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_host", default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=34235)
    parser.add_argument("--num_machines", type=int, required=True)
    parser.add_argument("--num_gpus_per_machine", type=int, required=True)
    parser.add_argument("--num_outputs", type=int, default=1024)

    args = parser.parse_args()
    main(args)
