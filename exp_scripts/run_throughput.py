import argparse
import time

from common import (
    run_exp,
    turn_off_memory_cache,
    skip_oom_case,
    get_fanouts,
    ThroughputExpParams
)

def main(args):
    print(f"Start run_throughput.py args={args}", flush=True)
    start_t = time.time()
    gnn_names = ["sage"]
    graph_names = ["fb10b"]
    num_layers = 3

    exec_type = args.exec_type
    num_machines = args.num_machines
    num_gpus_per_machine = args.num_gpus_per_machine
    req_per_sec_list = [float(r) for r in args.req_per_sec_list.split(",")]

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")
    batch_size = 1024

    exp_result_dir = f"{args.exp_root_dir}/throughput"

    for graph_name in graph_names:
        for gnn in gnn_names:
            for req_per_sec in req_per_sec_list:
                if skip_oom_case(exec_type, gnn, graph_name):
                    continue

                run_exp(
                    num_machines=num_machines,
                    num_gpus_per_machine=num_gpus_per_machine,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=num_layers,
                    fanouts=get_fanouts(exec_type, num_layers),
                    exec_type=exec_type,
                    exp_type="throughput",
                    throughput_exp_params=ThroughputExpParams(
                        req_per_sec=req_per_sec,
                        exp_secs=args.exp_secs),
                    batch_size=batch_size,
                    exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}_rps_{req_per_sec}_nm_{num_machines}_ngpu_{num_gpus_per_machine}",
                    extra_env_names=extra_env_names,
                    force_cuda_mem_uncached=turn_off_memory_cache(exec_type, gnn, graph_name))

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--req_per_sec_list', type=str, required=True)
    parser.add_argument('--exp_secs', type=float, required=True)
    parser.add_argument('--exec_type', type=str, required=True)
    parser.add_argument('--num_machines', type=int, default=4)
    parser.add_argument('--num_gpus_per_machine', type=int, default=1)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
