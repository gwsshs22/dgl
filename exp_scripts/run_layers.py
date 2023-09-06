import argparse
import time

from common import run_exp, LatencyExpParams

def main(args):
    print(f"Start run_layers.py args={args}")
    start_t = time.time()

    graph_names = ["ogbn-products", "fb10b"]
    exec_types = ["cgp-multi", "cgp", "dp-precoms", "dp"]

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")

    gnn = "sage"
    batch_size = 1024
    exp_result_dir = f"{args.exp_root_dir}/layers"
    for graph_name in graph_names:
        for fanouts in [[5, 10], [5, 10, 15], [5, 10, 15, 20]]:
            for exec_type in exec_types:
                fanout_str = "_".join([str(f) for f in fanouts])
                num_layers = len(fanouts)
                run_exp(
                    num_machines=4,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=num_layers,
                    fanouts=fanouts,
                    exec_type=exec_type,
                    exp_type="latency",
                    latency_exp_params=LatencyExpParams(num_reqs=args.num_reqs),
                    batch_size=batch_size,
                    exp_result_dir=f"{exp_result_dir}/{graph_name}_{fanout_str}_{exec_type}_sampled",
                    extra_env_names=extra_env_names
                )

                full_infer_num_reqs = args.num_reqs
                if exec_type == "dp":
                    full_infer_num_reqs = args.full_dp_num_reqs


    for graph_name in graph_names:
        for num_layers in [2, 3, 4]:
            for exec_type in exec_types:
                fanout_str = "_".join(["0"] * num_layers)

                full_infer_num_reqs = args.num_reqs
                if exec_type == "dp":
                    full_infer_num_reqs = args.full_dp_num_reqs

                run_exp(
                    num_machines=4,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=num_layers,
                    fanouts=None,
                    exec_type=exec_type,
                    exp_type="latency",
                    latency_exp_params=LatencyExpParams(num_reqs=full_infer_num_reqs),
                    batch_size=batch_size,
                    exp_result_dir=f"{exp_result_dir}/{graph_name}_{fanout_str}_{exec_type}_full",
                    extra_env_names=extra_env_names
                )

    print(f"Total experiments time={time.time() - start_t}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--num_reqs', type=int, default=300)
    parser.add_argument('--full_dp_num_reqs', type=int, default=5)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
