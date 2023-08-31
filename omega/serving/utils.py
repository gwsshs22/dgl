import torch

def init_torch_distributed(
    exec_mode,
    num_machines,
    num_gpus_per_machine,
    master_ip,
    master_port,
    global_rank,
    local_rank,
    backend):

    world_size = num_machines * num_gpus_per_machine
    if exec_mode == "cgp" or exec_mode == "cgp-multi":
        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_ip}:{master_port}",
            world_size=world_size,
            rank=global_rank)

        if exec_mode == "cgp":
            gpu_ranks = [r for r in range(world_size)]
            dist_group = torch.distributed.new_group()
        else:
            for i in range(num_gpus_per_machine):
                ranks_in_group = [r for r in range(i, world_size + i, num_gpus_per_machine)]
                new_group = torch.distributed.new_group(ranks=ranks_in_group)
                if i == local_rank:
                    dist_group = new_group
                    gpu_ranks = ranks_in_group

        torch.distributed.barrier()
        return dist_group, gpu_ranks
    else:
        return None, [r for r in range(world_size)]
