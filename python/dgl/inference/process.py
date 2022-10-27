import argparse

from .._ffi.function import _init_api

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role")
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.role == "master":
        _CAPI_DGLInferenceExecMasterProcess()
    else:
        _CAPI_DGLInferenceExecWorkerProcess()

_init_api("dgl.inference.process")
