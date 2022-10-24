"""Inference server."""
from .._ffi.function import _init_api

def exec_master_process():
    _CAPI_DGLInferenceExecMasterProcess()

def exec_worker_process():
    _CAPI_DGLInferenceExecWorkerProcess()

_init_api("dgl.inference.inference_server")
