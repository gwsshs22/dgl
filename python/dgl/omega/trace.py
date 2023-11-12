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
            start_time = time.perf_counter()
        yield
    finally:
        if TRACE_ENABLED:
            if device:
                torch.cuda.synchronize(device)
            TRACES.append((batch_id, name, int((time.perf_counter() - start_time) * 1000000)))

def trace_blocks(batch_id, blocks):
    global TRACE_ENABLED
    if TRACE_ENABLED:
        for layer_idx, block in enumerate(blocks):
            TRACES.append((batch_id, f"block{layer_idx}_num_srcs", block.num_src_nodes()))
            TRACES.append((batch_id, f"block{layer_idx}_num_dsts", block.num_dst_nodes()))
            TRACES.append((batch_id, f"block{layer_idx}_num_edges", block.num_edges()))

def collect_dist_block_stats(batch_id, blocks):
    global TRACE_ENABLED
    if TRACE_ENABLED:
        for layer_idx, block in enumerate(blocks):
            all_gather_time = 0 
            all_gather_size = 0
            for se, ee, size in block.all_gather_tracing:
                all_gather_time += int(se.elapsed_time(ee) * 1000)
                all_gather_size += size

            TRACES.append((batch_id, f"block{layer_idx}_all_gather", all_gather_time))
            TRACES.append((batch_id, f"block{layer_idx}_all_gather_size", all_gather_size))

            all_to_all_time = 0 
            all_to_all_size = 0
            for se, ee, size in block.all_to_all_tracing:
                all_to_all_time += int(se.elapsed_time(ee) * 1000)
                all_to_all_size += size

            TRACES.append((batch_id, f"block{layer_idx}_all_to_all", all_to_all_time))
            TRACES.append((batch_id, f"block{layer_idx}_all_to_all_size", all_to_all_size))


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
