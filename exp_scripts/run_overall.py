import argparse
import time

from common import (
    run_exp,
    turn_off_memory_cache,
    skip_oom_case,
    get_fanouts,
    LatencyExpParams
)

def main(args):
    print(f"Start run_overall.py args={args}", flush=True)
    start_t = time.time()

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")

    gnn_names = ["gcn", "sage", "gat"]
    graph_names = ["yelp", "amazon", "fb10b", "reddit", "ogbn-products", "ogbn-papers100M"]
    exec_types = ["cgp-multi", "dp-precoms", "dp-sampled", "dp"]

    exp_result_dir = f"{args.exp_root_dir}/overall"
    batch_size = 1024

    for graph_name in graph_names:
        for gnn in gnn_names:
            for exec_type in exec_types:        
                if skip_oom_case(exec_type, gnn, graph_name):
                    continue
                num_reqs = args.num_reqs if exec_type != "dp" else args.full_dp_num_reqs
                num_layers = 2 if gnn == "gcn" else 3
                run_exp(
                    num_machines=4,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=num_layers,
                    fanouts=get_fanouts(exec_type, num_layers),
                    exec_type=exec_type,
                    exp_type="latency",
                    latency_exp_params=LatencyExpParams(num_reqs=num_reqs),
                    batch_size=batch_size,
                    exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}",
                    extra_env_names=extra_env_names,
                    force_cuda_mem_uncached=turn_off_memory_cache(exec_type, gnn, graph_name))

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--num_reqs', type=int, default=500)
    parser.add_argument('--full_dp_num_reqs', type=int, default=50)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
