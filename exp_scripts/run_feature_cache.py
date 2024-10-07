import argparse
import time

from common import (
    run_exp,
    turn_off_memory_cache,
    skip_oom_case,
    get_fanouts,
    LatencyExpParams
)

def gb_to_cache_size(gb, feature_dim=1024, dtype_size=4):
    if gb is None:
        return None

    return int(gb * 1000000000 / (dtype_size * feature_dim))

def main(args):
    print(f"Start run_feature_cache.py args={args}", flush=True)
    start_t = time.time()
    gnn = "sage"
    num_layers = 3
    graph_name = "fb10b"
    exec_types = ["cgp-multi", "dp-sampled"]
    cache_sizes_in_gb = [None, 8, 4, 2, 1, 16]

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")
    batch_size = 1024

    exp_result_dir = f"{args.exp_root_dir}/feature_cache"

    for cache_size_in_gb in cache_sizes_in_gb:
        for exec_type in exec_types:
            if skip_oom_case(exec_type, gnn, graph_name):
                continue
            num_reqs = args.num_reqs
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
                exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}_cs_{cache_size_in_gb}gb",
                extra_env_names=extra_env_names,
                feature_cache_size=gb_to_cache_size(cache_size_in_gb),
                force_cuda_mem_uncached=turn_off_memory_cache(exec_type, gnn, graph_name)
            )

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--num_reqs', type=int, default=500)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
