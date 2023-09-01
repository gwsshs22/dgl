import argparse
import threading
import os
import sys
import json
from pathlib import Path

import torch
import torch.distributed.rpc as rpc
import numpy as np

import dgl
from dgl.distributed.dist_context import get_dist_graph_server
from dgl.sampling.neighbor import sample_edges
import dgl.backend as F

def dgl_server_main(
    ip_config,
    net_type,
    feature_dim,
    use_precoms,
    num_layers,
    num_hiddens,
    precom_path,
    random_seed):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    dgl.distributed.initialize(
        ip_config,
        net_type=net_type,
        feature_dim=feature_dim,
        load_precoms=use_precoms,
        num_layers=num_layers,
        num_hiddens=num_hiddens,
        precom_path=precom_path)

class OmegaGraphServer:

    def __init__(self, machine_rank, part_config, use_precoms, num_layers):
        dist_graph_server = get_dist_graph_server()
        self._local_data_store = {}
        self._local_g = dist_graph_server.local_partition

        key_names = ["features"]

        if use_precoms:
            for layer_idx in range(num_layers - 1):
                key_names.append(f"layer_{layer_idx}")

        for key in key_names:
            assert f"node~_N~{key}" in dist_graph_server.data_store
            self._local_data_store[key] = dist_graph_server.data_store[f"node~_N~{key}"]


    def remote_pull(self, name, local_nids):
        return self._local_data_store[name][local_nids]
    
    def remote_sampling(self, seeds, fanout):
        return sample_edges(self._local_g, seeds, fanout)

def main(args):
    dgl_server_thread = threading.Thread(
        target=dgl_server_main,
        args=(
            args.ip_config,
            args.net_type,
            args.feature_dim,
            args.use_precoms,
            args.num_layers,
            args.num_hiddens,
            args.precom_path,
            args.random_seed))
    dgl_server_thread.start()

    num_omega_groups = args.num_omega_groups

    num_machines = args.num_machines
    machine_rank = args.machine_rank
    num_gpus_per_machine = args.num_gpus_per_machine

    world_size = num_machines * num_gpus_per_machine
    server_rank = machine_rank

    os.environ["MASTER_ADDR"] = str(args.master_ip)
    os.environ["MASTER_PORT"] = str(args.master_rpc_port)

    rpc.init_rpc(f"server-{server_rank}",
        rank=world_size * num_omega_groups + 1 + server_rank,
        world_size=world_size * num_omega_groups + 1 + num_machines
    )
    rpc.shutdown()
    dgl_server_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--master_rpc_port', type=int)
    parser.add_argument("--num_omega_groups", type=int, default=1)
    parser.add_argument('--num_machines', type=int)
    parser.add_argument('--machine_rank', type=int)
    parser.add_argument('--num_gpus_per_machine', type=int)
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--net_type', type=str, default='socket',
                        help="backend net type, 'socket' or 'tensorpipe'")
    parser.add_argument('--feature_dim', type=int)
    parser.add_argument("--use_precoms", action="store_true")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_hiddens", type=int)
    parser.add_argument("--precom_path", type=str, default="")

    parser.add_argument('--random_seed', type=int, default=5123412)
    args = parser.parse_args()

    main(args)
