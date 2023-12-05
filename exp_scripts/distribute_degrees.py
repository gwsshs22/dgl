import argparse
import os
import sys

def main(args):
    num_parts = args.num_parts
    graph_partitioning = args.graph_partitioning
    machine_ips = args.machine_ips.split(",")[:num_parts]

    for i, part_id in enumerate(range(num_parts)):
        if i == args.my_rank:
            continue

        target_path = f"$DGL_DATA_HOME/omega_datasets-{num_parts}/{args.dataset}-{graph_partitioning}-{num_parts}/degrees.dgl"
        os.system(f"rsync -avP --mkpath {target_path} {args.ssh_username}@{machine_ips[i]}:{target_path}")
        target_path = f"$DGL_DATA_HOME/omega_datasets-{num_parts}/{args.dataset}-{graph_partitioning}-{num_parts}-outedges/degrees.dgl"
        os.system(f"rsync -avP --mkpath {target_path} {args.ssh_username}@{machine_ips[i]}:{target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_parts', type=int)
    parser.add_argument('--machine_ips', type=str)
    parser.add_argument('--my_rank', type=int, default=-1)
    parser.add_argument('--graph_partitioning', type=str, default='random')
    parser.add_argument('--ssh_username', type=str, default='gwkim')

    args = parser.parse_args()
    main(args)
