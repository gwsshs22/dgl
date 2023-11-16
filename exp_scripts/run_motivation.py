import argparse
import time
import os

from common import run_exp, LatencyExpParams

def main(args):
    print(f"Start run_overall_latency.py args={args}", flush=True)

    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    start_t = time.time()
    gnn = "gat"
    num_layers = 3

    graph_names = ["yelp", "amazon"]

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")
    batch_size = 1024

    exp_result_dir = f"{args.exp_root_dir}/motivation"

    for graph_name in graph_names:
        run_exp(
            num_machines=4,
            graph_name=graph_name,
            gnn=gnn,
            num_layers=num_layers,
            fanouts=[5, 10, 15],
            exec_type="dp",
            exp_type="latency",
            latency_exp_params=LatencyExpParams(num_reqs=args.num_reqs),
            batch_size=batch_size,
            exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_dp_sampled",
            extra_env_names=extra_env_names,
            recom_threshold=100)

        run_exp(
            num_machines=4,
            graph_name=graph_name,
            gnn=gnn,
            num_layers=num_layers,
            fanouts=[],
            exec_type="dp",
            exp_type="latency",
            latency_exp_params=LatencyExpParams(num_reqs=args.full_dp_num_reqs),
            batch_size=batch_size,
            exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_dp",
            extra_env_names=extra_env_names if graph_name == "yelp" else extra_env_names + ["PYTORCH_NO_CUDA_MEMORY_CACHING"],
            recom_threshold=100)

        for exec_type in ["cgp-multi", "cgp", "dp-precoms"]:
            num_reqs = args.num_reqs
            run_exp(
                num_machines=4,
                graph_name=graph_name,
                gnn=gnn,
                num_layers=num_layers,
                fanouts=[],
                exec_type=exec_type,
                exp_type="latency",
                latency_exp_params=LatencyExpParams(num_reqs=num_reqs),
                batch_size=batch_size,
                exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}_norecom",
                extra_env_names=extra_env_names,
                recom_threshold=100)

            run_exp(
                num_machines=4,
                graph_name=graph_name,
                gnn=gnn,
                num_layers=num_layers,
                fanouts=[],
                exec_type=exec_type,
                exp_type="latency",
                latency_exp_params=LatencyExpParams(num_reqs=num_reqs),
                batch_size=batch_size,
                exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}_recom",
                extra_env_names=extra_env_names,
                recom_threshold='auto')

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--full_dp_num_reqs', type=int, default=50)
    parser.add_argument('--num_reqs', type=int, default=500)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
