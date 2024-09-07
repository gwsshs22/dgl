import argparse
import time

from common import run_exp, ThroughputExpParams

def main(args):
    print(f"Start run_throughput.py args={args}", flush=True)
    if args.sampled:
        assert args.exec_type == "dp"

    start_t = time.time()
    gnn_names = ["sage"]
    graph_names = ["fb10b"]

    exec_type = args.exec_type
    req_per_sec_list = [float(r) for r in args.req_per_sec_list.split(",")]

    extra_env_names = []
    if args.extra_env_names:
        extra_env_names = args.extra_env_names.split(",")
    batch_size = 1024

    exp_result_dir = f"{args.exp_root_dir}/throughput"

    for graph_name in graph_names:
        for gnn in gnn_names:
            for req_per_sec in req_per_sec_list:
                exp_dir = f"{exp_result_dir}/{graph_name}_{gnn}_{exec_type}"
                if args.sampled:
                    exp_dir += "_sampled"
                    fanouts = [5, 10, 15]
                else:
                    fanouts = []

                exp_dir += f"_rps_{req_per_sec}"
                run_exp(
                    num_machines=4,
                    num_gpus_per_machine=1,
                    graph_name=graph_name,
                    gnn=gnn,
                    num_layers=3,
                    fanouts=fanouts,
                    exec_type=exec_type,
                    exp_type="throughput", throughput_exp_params=ThroughputExpParams(
                        req_per_sec=req_per_sec,
                        exp_secs=args.exp_secs),
                    batch_size=batch_size,
                    exp_result_dir=exp_dir,
                    extra_env_names=extra_env_names,
                    recom_threshold=100)

    print(f"Total experiments time={time.time() - start_t}s", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', required=True)
    parser.add_argument('--req_per_sec_list', type=str, required=True)
    parser.add_argument('--exp_secs', type=float, required=True)
    parser.add_argument('--exec_type', type=str, required=True)
    parser.add_argument('--sampled', action="store_true")
    parser.add_argument('--extra_env_names', type=str, default="")

    args = parser.parse_args()
    main(args)
