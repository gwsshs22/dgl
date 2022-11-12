import argparse
import os

import random
import numpy as np
import torch

import dgl.inference.envs as envs
from .models.gcn import DistGCN
from .models.sage import DistSAGE
from .models.gat import DistGAT
from .graph_server_process import GraphServerProcess
from .gnn_executor_process import GnnExecutorProcess
from .sampler_process import SamplerProcess
from .._ffi.function import _init_api
from .._ffi.object import register_object, ObjectBase

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["master", "worker"])
    parser.add_argument("--master-host", required=True)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--node-rank", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-devices-per-node", type=int,  required=True)
    parser.add_argument("--ip-config-path", required=True)
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--graph-config-path", required=True)
    parser.add_argument("--iface", type=str, required=False, default="")
    parser.add_argument('--random_seed', type=int, default=412412322)

    parser.add_argument("--parallelization-type", type=str, required=True, choices=["data", "p3", "vcut"])
    parser.add_argument("--using-precomputed-aggregations", action="store_true")

    # Model parameters
    parser.add_argument("--model", type=str, required=True, choices=["gcn", "sage", "gat"])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_inputs', type=int, default=32)
    parser.add_argument('--num_hiddens', type=int, default=32)
    parser.add_argument('--num_outputs', type=int, default=32)
    parser.add_argument('--heads', type=str, default="8,8", help="The number of attention heads for two-layer gat models")

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
    os.environ[envs.DGL_INFER_GRAPH_NAME] = args.graph_name
    os.environ[envs.DGL_INFER_GRAPH_CONFIG_PATH] = args.graph_config_path
    os.environ[envs.DGL_INFER_IFACE] = args.iface
    os.environ[envs.DGL_INFER_RANDOM_SEED] = str(args.random_seed)

    os.environ[envs.DGL_INFER_MODEL_TYPE] = args.model
    os.environ[envs.DGL_INFER_NUM_LAYERS] = str(args.num_layers)
    os.environ[envs.DGL_INFER_NUM_INPUTS] = str(args.num_inputs)
    os.environ[envs.DGL_INFER_NUM_HIDDENS] = str(args.num_hiddens)
    os.environ[envs.DGL_INFER_NUM_OUTPUTS] = str(args.num_outputs)
    os.environ[envs.DGL_INFER_HEADS] = args.heads

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
    graph_name = os.environ[envs.DGL_INFER_GRAPH_NAME]
    graph_config_path = os.environ[envs.DGL_INFER_GRAPH_CONFIG_PATH]
    iface = os.environ[envs.DGL_INFER_IFACE]
    random_seed = int(os.environ[envs.DGL_INFER_RANDOM_SEED])
    parallel_type = envs.get_parallelization_type()

    model_type = os.environ[envs.DGL_INFER_MODEL_TYPE]
    num_layers = int(os.environ[envs.DGL_INFER_NUM_LAYERS])
    num_inputs = int(os.environ[envs.DGL_INFER_NUM_INPUTS])
    num_hiddens = int(os.environ[envs.DGL_INFER_NUM_HIDDENS])
    num_outputs = int(os.environ[envs.DGL_INFER_NUM_OUTPUTS])
    heads = os.environ[envs.DGL_INFER_HEADS]

    channel = ActorProcessChannel()

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    if num_layers != 2:
        print(f"Currently only num_layers = 2 is allowed. Given={num_layers}")
        exit(-1)

    os.environ["DGL_DIST_MODE"] = "distributed"

    if actor_process_role == "gnn_executor":
        if model_type == 'gcn':
            model = DistGCN(num_inputs, num_hiddens, num_outputs, num_layers)
        elif model_type == 'sage':
            model = DistSAGE(num_inputs, num_hiddens, num_outputs, num_layers)
        elif model_type == 'gat':
            heads = list(map(lambda x: int(x), heads.split(",")))
            model = DistGAT(num_inputs, num_hiddens, num_outputs, num_layers, heads)
        else:
            print(f"Unknown model_type: {model_type}")
            exit(-1)
        actor_process = GnnExecutorProcess(channel, num_nodes, ip_config_path, parallel_type, graph_name, graph_config_path, local_rank, model)
    elif actor_process_role == "graph_server":
        actor_process = GraphServerProcess(channel, num_nodes, node_rank, num_devices_per_node, ip_config_path, graph_config_path, parallel_type)
    elif actor_process_role == "sampler":
        actor_process = SamplerProcess(channel, num_nodes, ip_config_path, graph_name, graph_config_path, local_rank)
    else:
        print(f"Unknown actor_process_role: {actor_process_role}")
        exit(-1)

    actor_process.run()

class ActorProcessChannel:

    def notify_initialized(self):
        _CAPI_DGLInferenceActorNotifyInitialized()

    def fetch_request(self):
        return _CAPI_DGLInferenceActorFetchRequest()

@register_object('inference.process.ActorRequest')
class ActorRequest(ObjectBase):

    @property
    def request_type(self):
        return _CAPI_DGLInferenceActorRequestGetRequestType(self)

    @property
    def batch_id(self):
        return _CAPI_DGLInferenceActorRequestGetBatchId(self)
    
    def done(self):
        _CAPI_DGLInferenceActorRequestDone(self)

_init_api("dgl.inference.process")
