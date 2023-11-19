import argparse
import time

from common import run_exp, LatencyExpParams

def main(args):
    print(f"Start run_feature_dims.py args={args}", flush=True)
    start_t = time.time()

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")

    graph_names = ["ogbn-products", "fb10b"]
    exec_types = ["cgp-multi", "cgp", "dp-precoms"]
    gnn = "sage"
    num_layers = 3
    exp_result_dir = f"{args.exp_root_dir}/feature_dims"

    for graph_name in graph_names:
        for feature_dim in [64, 128, 256, 512, 1024, 2048]:
            for exec_type in exec_types:
                num_reqs = args.num_reqs if exec_type != "dp" else args.full_dp_num_reqs

                force_cuda_mem_uncached = graph_name == "fb10b" and exec_type == "dp"
                run_exp(
                    num_machines=4,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=num_layers,
                    fanouts=[],
                    exec_type=exec_type,
                    exp_type="latency",
                    latency_exp_params=LatencyExpParams(num_reqs=num_reqs),
                    feature_dim=feature_dim,
                    exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}_fd_{feature_dim}",
                    extra_env_names=extra_env_names,
                    recom_threshold=100,
                    force_cuda_mem_uncached=force_cuda_mem_uncached)

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--num_reqs', type=int, default=500)
    parser.add_argument('--full_dp_num_reqs', type=int, default=50)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
