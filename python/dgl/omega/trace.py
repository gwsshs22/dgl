import contextlib
import time
import os

from pathlib import Path

import torch
import os

TRACE_ENABLED = False
TRACES = []
def enable_tracing(breakdown_trace_dir):
    global TRACE_ENABLED
    if breakdown_trace_dir:
        TRACE_ENABLED = True
        os.makedirs(breakdown_trace_dir, exist_ok=True)
    


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

def write_traces(result_dir, file_name, num_warmups=None):
    if TRACE_ENABLED:
        print(f'write_traces to {Path(result_dir) / f"{file_name}.txt"}')
        with open(Path(result_dir) / f"{file_name}.txt", "a") as f:
            for batch_id, name, elapsed_micro in TRACES:
                f.write(f"{batch_id},{name},{elapsed_micro}\n")
        if num_warmups is not None:
            with open(Path(result_dir) / "num_warmups.txt", "a") as f:
                f.write(f"{num_warmups}\n")
