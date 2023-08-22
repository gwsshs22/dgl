from dataclasses import dataclass
from typing import Any
from pathlib import Path
import time

import torch
import numpy as np

import dgl

@dataclass(frozen=True)
class InferBatchRequest:
    target_gnids: Any
    target_features: Any
    src_gnids: Any
    dst_gnids: Any

class RequestGenerator:

    def __init__(self, trace_dir, req_per_sec):
        self._load_traces(trace_dir)
        self._req_per_sec = req_per_sec
        self._idx = 0

    def _load_traces(self, trace_dir):
        trace_dir = Path(trace_dir)
        self._num_traces = int((trace_dir / "num_traces.txt").read_text())
        self._batch_requests = []
        for i in range(self._num_traces):
            tensors = dgl.data.load_tensors(str(trace_dir / f"{i}.dgl"))
            self._batch_requests.append(InferBatchRequest(
                target_gnids=tensors["target_gnids"],
                target_features=tensors["target_features"],
                src_gnids=tensors["src_gnids"],
                dst_gnids=tensors["dst_gnids"]
            ))

    def __iter__(self):
        self._idx = 0
        return self

    def _sleep(self):
        if self._req_per_sec <= 0.0:
            return
        time.sleep(np.random.exponential(1 / self._req_per_sec))

    def __next__(self):
        self._idx += 1
        self._sleep()
        return self._batch_requests[self._idx % self._num_traces]


def create_req_generator(trace_dir, req_per_sec):
    return RequestGenerator(trace_dir, req_per_sec)
