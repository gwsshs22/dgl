import argparse
import time

from common import run_exp, LatencyExpParams

def main(args):
    print(f"Start run_hidden_dims.py args={args}", flush=True)
    start_t = time.time()

    graph_names = ["fb10b"]
    exec_types = ["cgp-multi", "cgp", "dp-precoms", "dp"]
    num_hiddens_list = [128, 256, 512, 1024, 2048, 4096]

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")

    batch_size = 1024
    gnn = "sage"
    exp_result_dir = f"{args.exp_root_dir}/hidden_dims"
    for graph_name in graph_names:
        for num_hiddens in num_hiddens_list:
            for exec_type in exec_types:
                run_exp(
                    num_machines=4,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=3,
                    fanouts=[5, 10, 15],
                    exec_type=exec_type,
                    exp_type="latency",
                    latency_exp_params=LatencyExpParams(num_reqs=args.num_reqs),
                    batch_size=batch_size,
                    num_hiddens=num_hiddens,
                    exp_result_dir=f"{exp_result_dir}/{graph_name}_{num_hiddens}_{exec_type}_sampled",
                    extra_env_names=extra_env_names
                )

                full_infer_num_reqs = args.num_reqs
                if exec_type == "dp":
                    full_infer_num_reqs = args.full_dp_num_reqs

                run_exp(
                    num_machines=4,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=3,
                    fanouts=None,
                    exec_type=exec_type,
                    exp_type="latency",
                    latency_exp_params=LatencyExpParams(num_reqs=full_infer_num_reqs),
                    batch_size=batch_size,
                    num_hiddens=num_hiddens,
                    exp_result_dir=f"{exp_result_dir}/{graph_name}_{num_hiddens}_{exec_type}_full",
                    extra_env_names=extra_env_names
                )
    
    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--num_reqs', type=int, default=300)
    parser.add_argument('--full_dp_num_reqs', type=int, default=5)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
