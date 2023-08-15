import argparse
import time

import torch
import torch.distributed as dist

from dgl.omega.dist_context import init_omega, receive_request


def main(args):
    num_machines = args.num_machines
    machine_rank = args.machine_rank
    num_gpus_per_machine = args.num_gpus_per_machine
    gpu_rank = args.gpu_rank

    init_omega(
        args.master_host,
        args.master_port,
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        gpu_rank)

    for _ in range(10):
        new_gnids, src_gnids, dst_gnids, new_features = receive_request(
            num_machines, machine_rank, num_gpus_per_machine, gpu_rank)
        
        t1 = time.time()
        output = torch.rand((new_features.shape[0], args.num_outputs))
        t2 = time.time()
        print(f"process={t2-t1:.4f}, new_features.shape={new_features.shape}")
        dist.isend(output, 0).wait()
        t3 = time.time()
        print(f"result send={t3-t2:.4f}")


    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_host", default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=34235)
    parser.add_argument("--num_machines", type=int, required=True)
    parser.add_argument("--machine_rank", type=int, required=True)
    parser.add_argument("--num_gpus_per_machine", type=int, required=True)
    parser.add_argument("--gpu_rank", type=int, required=True)
    parser.add_argument("--num_outputs", type=int, default=1024)

    args = parser.parse_args()
    main(args)
