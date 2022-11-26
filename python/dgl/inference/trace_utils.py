import contextlib
import time
import os

from pathlib import Path

TRACE_ENABLED = False
TRACES = []
def enable_tracing():
    global TRACE_ENABLED
    TRACE_ENABLED = True

@contextlib.contextmanager
def trace_me(batch_id, name):
    global TRACE_ENABLED
    try:
        if TRACE_ENABLED:
            start_time = time.time()
        yield
    finally:
        if TRACE_ENABLED:
            TRACES.append((batch_id, name, int((time.time() - start_time) * 1000000)))

def write_traces(result_dir, node_rank):
    print(f'write_traces to {Path(result_dir) / f"node_{node_rank}.txt"}')
    with open(Path(result_dir) / f"node_{node_rank}.txt", "a") as f:
        for batch_id, name, elapsed_micro in TRACES:
            f.write(f"{batch_id},{name},{elapsed_micro}\n")

