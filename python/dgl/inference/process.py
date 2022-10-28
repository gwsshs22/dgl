import argparse
import os

import dgl.inference.envs as envs
from .._ffi.function import _init_api

def make_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role")
    parser.add_argument("--master-host")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--node-rank", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-devices-per-node", type=int)
    parser.add_argument("--iface", type=str, required=False, default="")
    return parser

def main():
    parser = make_main_parser()
    args = parser.parse_args()

    if args.role == "master":
        if args.node_rank != 0:
            print(f"The node_rank of master should be 0 but {args.node_rank} is given.")
            exit(-1)
        node_rank = 0
    else:
        if args.node_rank <= 0:
            print(f"Invalid node_rank {args.node_rank} for a worker.")
            exit(-1)
        node_rank = args.node_rank

    # TODO: parameter validation
    os.environ[envs.DGL_INFER_MASTER_HOST] = args.master_host
    os.environ[envs.DGL_INFER_MASTER_PORT] = str(args.master_port)
    os.environ[envs.DGL_INFER_NODE_RANK] = str(node_rank)
    os.environ[envs.DGL_INFER_NUM_NODES] = str(args.num_nodes)
    os.environ[envs.DGL_INFER_NUM_DEVICES_PER_NODE] = str(args.num_devices_per_node)
    os.environ[envs.DGL_INFER_IFACE] = args.iface

    if args.role == "master":
        _CAPI_DGLInferenceExecMasterProcess()
    else:
        _CAPI_DGLInferenceExecWorkerProcess()

# Call by a new child actor process
def fork():
    _CAPI_DGLInferenceStartActorProcessThread()
    _CAPI_DGLInferenceActorNotifyInitialized()
    while True:
        _CAPI_DGLInferenceActorFetchRequest()

_init_api("dgl.inference.process")
