from dataclasses import dataclass
from typing import Any
from pathlib import Path
import threading
import time
import queue

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

    def __init__(self, trace_dir, req_per_sec, arrival_type, random_seed):
        self._load_traces(trace_dir)
        self._req_per_sec = req_per_sec
        self._arrival_type = arrival_type
        self._random_seed = random_seed
        self._num_reqs = None
        self._gen_tread = None
        self._queue = None

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

    def set_num_reqs(self, num_reqs):
        self._num_reqs = num_reqs

    @staticmethod
    def gen_thread_main(q, num_reqs, req_per_sec, arrival_type, random_seed):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        for idx in range(num_reqs):
            if req_per_sec > 0.0:
                if arrival_type == "poisson":
                    sleep_time = np.random.exponential(1 / req_per_sec)
                else:
                    sleep_time = 1 / req_per_sec
                time.sleep(sleep_time)
            q.put(idx)
        
        q.put(None)

    def __iter__(self):
        self._idx = 0
        assert self._num_reqs is not None
        assert self._gen_tread is None
        assert self._queue is None

        self._queue = queue.SimpleQueue()

        self._gen_thread = threading.Thread(
            target=RequestGenerator.gen_thread_main,
            args=(
                self._queue,
                self._num_reqs,
                self._req_per_sec,
                self._arrival_type,
                self._random_seed
            ),
            daemon=True
        )

        self._gen_thread.start()
        return self

    def __next__(self):
        idx = self._queue.get()
        if idx is None:
            self._num_reqs = None
            self._gen_thread = None
            self._queue = None
            raise StopIteration

        return self._batch_requests[idx % self._num_traces]


def create_req_generator(trace_dir, req_per_sec, arrival_type, random_seed):
    return RequestGenerator(trace_dir, req_per_sec, arrival_type, random_seed)
