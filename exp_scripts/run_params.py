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
    print(f"Start run_params.py args={args}", flush=True)
    start_t = time.time()

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")

    graph_names = ["fb10b"]
    exec_types = ["cgp-multi", "dp-sampled"]
    gnn = "sage"
    num_layers = 3
    num_reqs = args.num_reqs
    exp_result_dir = f"{args.exp_root_dir}/params"

    param_values = [64, 128, 256, 512, 1024, 2048]
    default_values = {
        "feature_size": 1024,
        "hidden_dims": 128,
        "batch_size": 1024
    }
    param_names = list(default_values.keys())

    for graph_name in graph_names:
        for param_name in param_names:
            for param_value in param_values:
                if param_value == default_values[param_name]:
                    continue
                
                exp_values = default_values.copy()
                exp_values[param_name] = param_value

                feature_size = exp_values["feature_size"]
                hidden_dims = exp_values["hidden_dims"]
                batch_size = exp_values["batch_size"]

                for exec_type in exec_types:
                    if skip_oom_case(exec_type, gnn, graph_name):
                        continue

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
                        feature_dim=feature_size,
                        num_hiddens=hidden_dims,
                        exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}_bs_{batch_size}_fs_{feature_size}_hd_{hidden_dims}",
                        extra_env_names=extra_env_names,
                        force_cuda_mem_uncached=turn_off_memory_cache(exec_type, gnn, graph_name))

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--num_reqs', type=int, default=500)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
