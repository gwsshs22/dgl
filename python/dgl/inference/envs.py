import os
from enum import Enum

DGL_INFER_MASTER_HOST = "DGL_INFER_MASTER_HOST"
DGL_INFER_MASTER_PORT = "DGL_INFER_MASTER_PORT"
DGL_INFER_MASTER_TORCH_PORT = "DGL_INFER_MASTER_TORCH_PORT"
DGL_INFER_NODE_RANK = "DGL_INFER_NODE_RANK"
DGL_INFER_LOCAL_RANK = "DGL_INFER_LOCAL_RANK"
DGL_INFER_NUM_NODES = "DGL_INFER_NUM_NODES"
DGL_INFER_NUM_DEVICES_PER_NODE = "DGL_INFER_NUM_DEVICES_PER_NODE"
DGL_INFER_IFACE = "DGL_INFER_IFACE"
DGL_INFER_RANDOM_SEED = "DGL_INFER_RANDOM_SEED"
DGL_INFER_ACTOR_PROCESS_ROLE = "DGL_INFER_ACTOR_PROCESS_ROLE"
DGL_INFER_ACTOR_PROCESS_GLOBAL_ID = "DGL_INFER_ACTOR_PROCESS_GLOBAL_ID"
DGL_INFER_IP_CONFIG_PATH = "DGL_INFER_IP_CONFIG_PATH"
DGL_INFER_GRAPH_NAME = "DGL_INFER_GRAPH_NAME"
DGL_INFER_GRAPH_CONFIG_PATH = "DGL_INFER_GRAPH_CONFIG_PATH"

DGL_INFER_PARALLELIZATION_TYPE = "DGL_INFER_PARALLELIZATION_TYPE"
DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS = "DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS"
DGL_INFER_MODEL_TYPE = "DGL_INFER_MODEL_TYPE"

DGL_INFER_MODEL_TYPE = "DGL_INFER_MODEL_TYPE"
DGL_INFER_NUM_LAYERS = "DGL_INFER_NUM_LAYERS"
DGL_INFER_NUM_INPUTS = "DGL_INFER_NUM_INPUTS"
DGL_INFER_NUM_HIDDENS = "DGL_INFER_NUM_HIDDENS"
DGL_INFER_NUM_OUTPUTS = "DGL_INFER_NUM_OUTPUTS"
DGL_INFER_HEADS = "DGL_INFER_HEADS"

class ParallelizationType(Enum):
    DATA = 0
    P3 = 1
    VERTEX_CUT = 2

def get_parallelization_type():
    ptype = int(os.environ[DGL_INFER_PARALLELIZATION_TYPE])
    if ptype == 0:
        return ParallelizationType.DATA
    elif ptype == 1:
        return ParallelizationType.P3
    else:
        return ParallelizationType.VERTEX_CUT
