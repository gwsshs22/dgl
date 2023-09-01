import contextlib
from dataclasses import dataclass
import time
import os

from pathlib import Path

import torch

from dgl.omega.omega_apis import collect_cpp_traces

TRACE_ENABLED = False
TRACES = []

@dataclass
class Trace:
    owner: str
    batch_id: int
    name: str
    elapsed_micro: int

def enable_tracing():
    global TRACE_ENABLED
    TRACE_ENABLED = True

@contextlib.contextmanager
def trace_me(batch_id, name, device=None):
    global TRACE_ENABLED
    try:
        if TRACE_ENABLED:
            start_time = time.time()
        yield
    finally:
        if TRACE_ENABLED:
            TRACES.append((batch_id, name, int((time.time() - start_time) * 1000000)))
            if device:
                torch.cuda.synchronize(device)

def put_trace(batch_id, name, t):
    if TRACE_ENABLED:
        TRACES.append((batch_id, name, int(t * 1000000)))

def get_traces(owner):
    return [Trace(owner, b, n, e) for b, n, e in TRACES]

def get_cpp_traces(owner):
    ret = collect_cpp_traces()
    ret_len = len(ret)
    traces = []
    assert ret_len % 3 == 0
    for i in range(ret_len // 3):
        traces.append(Trace(owner, ret[3 * i], ret[3 * i + 1], ret[3 * i + 2]))
    return traces
