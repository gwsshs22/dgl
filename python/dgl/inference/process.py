import argparse
import os

import random
import numpy as np
import torch

import dgl.inference.envs as envs
from dgl.distributed.rpc import register_sig_handler
from .models.factory import load_model
from .graph_server_process import GraphServerProcess
from .gnn_executor_process import GnnExecutorProcess
from .sampler_process import SamplerProcess
from .trace_utils import enable_tracing
from .._ffi.function import _init_api
from .._ffi.object import register_object, ObjectBase

def main():
    register_sig_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["master", "worker"])
    parser.add_argument("--master-host", required=True)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--master-torch-port", type=int)
    parser.add_argument("--node-rank", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-backup-servers", type=int, default=3)
    parser.add_argument("--num-devices-per-node", type=int, required=True)
    parser.add_argument("--num-samplers-per-node", type=int, required=True)
    parser.add_argument("--ip-config-path", required=True)
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--graph-config-path", required=True)
    parser.add_argument("--iface", type=str, required=False, default="")
    parser.add_argument('--random_seed', type=int, default=412412322)

    parser.add_argument("--parallelization-type", type=str, required=True, choices=["data", "p3", "vcut"])
    parser.add_argument("--using-precomputed-aggregations", action="store_true")
    parser.add_argument("--precom-filename", type=str, default="")

    # Model parameters
    parser.add_argument("--model", type=str, required=True, choices=["gcn", "sage", "gat"])
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-inputs', type=int, default=32)
    parser.add_argument('--num-hiddens', type=int, default=32)
    parser.add_argument('--num-outputs', type=int, default=32)
    parser.add_argument('--heads', type=str, default="8,8", help="The number of attention heads for two-layer gat models")

    parser.add_argument('--input-trace-dir', type=str, default="")
    parser.add_argument('--num-warmups', type=int, default=32)
    parser.add_argument('--num-requests', type=int, default=258)
    parser.add_argument('--result-dir', type=str, default="")
    parser.add_argument('--collect-stats', action='store_true')
    parser.add_argument('--execute-one-by-one', action='store_true')

    args = parser.parse_args()

    if args.role == "master":
        if args.node_rank != 0:
            print(f"The node_rank of master should be 0 but {args.node_rank} is given.")
            exit(-1)
        node_rank = 0
        assert(args.input_trace_dir)
    else:
        if args.node_rank <= 0:
            print(f"Invalid node_rank {args.node_rank} for a worker.")
            exit(-1)
        node_rank = args.node_rank

    assert(not args.using_precomputed_aggregations or args.precom_filename != "")

    # TODO: parameter validation
    os.environ[envs.DGL_INFER_MASTER_HOST] = args.master_host
    os.environ[envs.DGL_INFER_MASTER_PORT] = str(args.master_port)
    os.environ[envs.DGL_INFER_MASTER_TORCH_PORT] = str(args.master_torch_port)
    os.environ[envs.DGL_INFER_NODE_RANK] = str(node_rank)
    os.environ[envs.DGL_INFER_NUM_BACKUP_SERVERS] = str(args.num_backup_servers)
    os.environ[envs.DGL_INFER_NUM_NODES] = str(args.num_nodes)
    os.environ[envs.DGL_INFER_NUM_DEVICES_PER_NODE] = str(args.num_devices_per_node)
    os.environ[envs.DGL_INFER_NUM_SAMPLERS_PER_NODE] = str(args.num_samplers_per_node)
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

    os.environ[envs.DGL_INFER_INPUT_TRACE_DIR] = args.input_trace_dir
    os.environ[envs.DGL_INFER_NUM_WARMUPS] = str(args.num_warmups)
    os.environ[envs.DGL_INFER_NUM_REQUESTS] = str(args.num_requests)

    os.environ[envs.DGL_INFER_RESULT_DIR] = args.result_dir
    os.environ[envs.DGL_INFER_COLLECT_STATS] = "1" if args.collect_stats else "0"
    os.environ[envs.DGL_INFER_EXECUTE_ONE_BY_ONE] = "1" if args.execute_one_by_one else "0"

    os.makedirs(args.result_dir, exist_ok=True)

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
    os.environ[envs.DGL_INFER_PRECOM_FILENAME] = args.precom_filename

    if args.role == "master":
        _CAPI_DGLInferenceExecMasterProcess()
    else:
        _CAPI_DGLInferenceExecWorkerProcess()

# Called by a new child actor process
def fork():
    register_sig_handler()
    _CAPI_DGLInferenceStartActorProcessThread()

    actor_process_role = os.environ[envs.DGL_INFER_ACTOR_PROCESS_ROLE]
    num_nodes = int(os.environ[envs.DGL_INFER_NUM_NODES])
    num_backup_servers = int(os.environ[envs.DGL_INFER_NUM_BACKUP_SERVERS])
    node_rank = int(os.environ[envs.DGL_INFER_NODE_RANK])
    local_rank = int(os.environ[envs.DGL_INFER_LOCAL_RANK])
    master_host = os.environ[envs.DGL_INFER_MASTER_HOST]
    master_torch_port = int(os.environ[envs.DGL_INFER_MASTER_TORCH_PORT])
    num_devices_per_node = int(os.environ[envs.DGL_INFER_NUM_DEVICES_PER_NODE])
    num_samplers_per_node = int(os.environ[envs.DGL_INFER_NUM_SAMPLERS_PER_NODE])
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

    using_precomputed_aggregations = envs.get_using_precomputed_aggregations()
    precom_filename = os.environ[envs.DGL_INFER_PRECOM_FILENAME]

    result_dir = os.environ[envs.DGL_INFER_RESULT_DIR]
    collect_stats = envs.get_collect_stats()

    if collect_stats:
        enable_tracing()

    channel = ActorProcessChannel()

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    if num_layers != 2:
        print(f"Currently only num_layers = 2 is allowed. Given={num_layers}")
        exit(-1)

    os.environ["DGL_DIST_MODE"] = "distributed"

    if actor_process_role == "gnn_executor":
        model = load_model(model_type, num_inputs, num_hiddens, num_outputs, num_layers, heads)
        actor_process = GnnExecutorProcess(channel,
                                           num_nodes,
                                           num_backup_servers,
                                           node_rank,
                                           num_devices_per_node,
                                           local_rank,
                                           master_host,
                                           master_torch_port,
                                           ip_config_path,
                                           parallel_type,
                                           using_precomputed_aggregations,
                                           graph_name,
                                           graph_config_path,
                                           model,
                                           num_inputs,
                                           result_dir,
                                           collect_stats)
    elif actor_process_role == "graph_server":
        actor_process = GraphServerProcess(channel,
                                           num_nodes,
                                           num_backup_servers,
                                           node_rank,
                                           num_devices_per_node,
                                           num_samplers_per_node,
                                           local_rank,
                                           ip_config_path,
                                           graph_config_path,
                                           parallel_type,
                                           using_precomputed_aggregations,
                                           precom_filename)
    elif actor_process_role == "sampler":
        actor_process = SamplerProcess(channel,
                                       num_nodes,
                                       num_backup_servers,
                                       node_rank,
                                       num_devices_per_node,
                                       local_rank,
                                       master_host,
                                       ip_config_path,
                                       parallel_type,
                                       using_precomputed_aggregations,
                                       graph_name,
                                       graph_config_path,
                                       result_dir)
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
    
    @property
    def param0(self):
        return _CAPI_DGLInferenceActorRequestGetParam0(self)

    def done(self):
        _CAPI_DGLInferenceActorRequestDone(self)

_init_api("dgl.inference.process")
