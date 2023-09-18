import argparse
import os
import sys

def main(args):
    num_parts = args.num_parts
    machine_ips = args.machine_ips.split(",")[:num_parts]
    if args.rsync_trace:
        os.system(f"rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/ {args.ssh_username}@{machine_ips[0]}:$DGL_DATA_HOME/omega_traces/")
    for i, part_id in enumerate(range(num_parts)):
        target_path = f"$DGL_DATA_HOME/omega_datasets-{num_parts}/{args.dataset}-random-{num_parts}/{args.dataset}.json"
        os.system(f"rsync -avP --mkpath {target_path} {args.ssh_username}@{machine_ips[i]}:{target_path}")
        target_path = f"$DGL_DATA_HOME/omega_datasets-{num_parts}/{args.dataset}-random-{num_parts}/part{i}/"
        os.system(f"rsync -avP --mkpath {target_path} {args.ssh_username}@{machine_ips[i]}:{target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_parts', type=int)
    parser.add_argument('--machine_ips', type=str)
    parser.add_argument('--rsync_trace', action="store_true")
    parser.add_argument('--ssh_username', type=str, default='gwkim')

    args = parser.parse_args()
    main(args)
