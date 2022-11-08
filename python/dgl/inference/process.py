import argparse
import os

import dgl.inference.envs as envs
from .graph_server_process import GraphServerProcess
from .gnn_executor_process import GnnExecutorProcess
from .sampler_process import SamplerProcess
from .._ffi.function import _init_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["master", "worker"])
    parser.add_argument("--master-host", required=True)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--node-rank", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-devices-per-node", type=int,  required=True)
    parser.add_argument("--ip-config-path", required=True)
    parser.add_argument("--graph-config-path", required=True)
    parser.add_argument("--iface", type=str, required=False, default="")
    parser.add_argument("--parallelization-type", type=str, required=True, choices=["data", "p3", "vcut"])
    parser.add_argument("--using-precomputed-aggregations", action="store_true")
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
    os.environ[envs.DGL_INFER_IP_CONFIG_PATH] = args.ip_config_path
    os.environ[envs.DGL_INFER_GRAPH_CONFIG_PATH] = args.graph_config_path
    os.environ[envs.DGL_INFER_IFACE] = args.iface

    if args.parallelization_type == "data":
        os.environ[envs.DGL_INFER_PARALLELIZATION_TYPE] = str(envs.ParallelizationType.DATA.value)
    elif args.parallelization_type == "p3":
        os.environ[envs.DGL_INFER_PARALLELIZATION_TYPE] = str(envs.ParallelizationType.P3.value)
    elif args.parallelization_type == "vcut":
        os.environ[envs.DGL_INFER_PARALLELIZATION_TYPE] = str(envs.ParallelizationType.VERTEX_CUT.value)
    else:
        print(f"Unexpected --parallelization-type {args.parallelization_type}")
        exit(-1)

    os.environ[envs.DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS] = "1" if args.using_precomputed_aggregations else "0"

    if args.role == "master":
        _CAPI_DGLInferenceExecMasterProcess()
    else:
        _CAPI_DGLInferenceExecWorkerProcess()

# Called by a new child actor process
def fork():
    _CAPI_DGLInferenceStartActorProcessThread()

    actor_process_role = os.environ[envs.DGL_INFER_ACTOR_PROCESS_ROLE]
    num_nodes = int(os.environ[envs.DGL_INFER_NUM_NODES])
    node_rank = int(os.environ[envs.DGL_INFER_NODE_RANK])
    local_rank = int(os.environ[envs.DGL_INFER_LOCAL_RANK])
    num_devices_per_node = int(os.environ[envs.DGL_INFER_NUM_DEVICES_PER_NODE])
    ip_config_path = os.environ[envs.DGL_INFER_IP_CONFIG_PATH]
    graph_config_path = os.environ[envs.DGL_INFER_GRAPH_CONFIG_PATH]
    iface = os.environ[envs.DGL_INFER_IFACE]
    parallel_type = envs.get_parallelization_type()
    channel = ActorProcessChannel()

    if actor_process_role == "gnn_executor":
        actor_process = GnnExecutorProcess(channel, num_nodes, ip_config_path)
    elif actor_process_role == "graph_server":
        actor_process = GraphServerProcess(channel, num_nodes, node_rank, num_devices_per_node, ip_config_path, graph_config_path, parallel_type)
    elif actor_process_role == "sampler":
        actor_process = SamplerProcess(channel, num_nodes, ip_config_path)
    else:
        printf(f"Unknown actor_process_role: {actor_process_role}")
        exit(-1)

    actor_process.run()

class ActorProcessChannel:

    def notify_initialized(self):
        _CAPI_DGLInferenceActorNotifyInitialized()

    def fetch_request(self):
        return _CAPI_DGLInferenceActorFetchRequest()

_init_api("dgl.inference.process")
