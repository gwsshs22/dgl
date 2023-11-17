import argparse
import time

from common import run_exp, LatencyExpParams

def turn_off_memory_cache(exec_type, graph_name, gnn, num_layers):
    if exec_type != "dp":
        return False

    if graph_name == "fb5b" or graph_name == "fb10b" or graph_name == "amazon":
        return True
    if (graph_name == "ogbn-products" or graph_name == "reddit") and gnn == "gat" and num_layers >= 3:
        return True

    return False


def main(args):
    print(f"Start run_scalability.py args={args}", flush=True)
    start_t = time.time()

    gnn_names = ["gcn", "sage", "gat"]
    graph_names = ["fb10b", "ogbn-products"]
    exec_types = ["cgp-multi", "cgp", "dp-precoms", "dp"]

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")
    batch_size = 1024

    exp_result_dir = f"{args.exp_root_dir}/scalability"

    for graph_name in graph_names:
        for num_machines in [3, 2]:
            for gnn in gnn_names:
                for exec_type in exec_types:
                    num_reqs = args.num_reqs
                    num_layers = 2 if gnn == "gcn" else 3

                    run_exp(
                        num_machines=num_machines,
                        graph_name=graph_name,
                        gnn=gnn,
                        num_layers=num_layers,
                        fanouts=[],
                        exec_type=exec_type,
                        exp_type="latency",
                        latency_exp_params=LatencyExpParams(num_reqs=num_reqs),
                        batch_size=batch_size,
                        exp_result_dir=f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}_nm_{num_machines}",
                        extra_env_names=extra_env_names,
                        recom_threshold='auto',
                        force_cuda_mem_uncached=turn_off_memory_cache(exec_type, graph_name, gnn, num_layers)
                    )

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--num_reqs', type=int, default=500)
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
