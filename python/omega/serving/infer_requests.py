import time
import threading
import queue

import torch
import numpy as np

import dgl

from omega.utils import load_traces

class RequestGenerator:

    def __init__(self, trace_dir, req_per_sec, arrival_type, random_seed, feature_dim):
        self._req_per_sec = req_per_sec
        self._arrival_type = arrival_type
        self._random_seed = random_seed
        self._feature_dim = feature_dim

        self._batch_requests = load_traces(
            trace_dir,
            feature_dim
        )
        self._num_traces = len(self._batch_requests)
        self._batch_size = self._batch_requests[0].target_gnids.shape[0]

        self._num_reqs = None
        self._req_gen_thread = None
        self._queue = None

    @staticmethod
    def req_gen_thread_main(q, num_reqs, req_per_sec, arrival_type, random_seed):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        for idx in range(num_reqs):
            q.put(idx)
            if req_per_sec <= 0.0:
                continue
            
            if arrival_type == "poisson":
                sleep_time = np.random.exponential(1 / req_per_sec)
            else:
                sleep_time = 1 / req_per_sec
            time.sleep(sleep_time)

        q.put(None)


    def set_num_reqs(self, num_reqs):
        self._num_reqs = num_reqs
    
    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        self._queue = queue.SimpleQueue()
        self._req_gen_thread = threading.Thread(
            target=RequestGenerator.req_gen_thread_main,
            args=(
                self._queue,
                self._num_reqs,
                self._req_per_sec,
                self._arrival_type,
                self._random_seed
            ),
            daemon=True
        )
        self._req_gen_thread.start()
        return self

    def __next__(self):
        idx = self._queue.get()
        if idx is None:
            self._queue = None
            self._num_reqs = None
            self._req_gen_thread = None
            raise StopIteration
        
        return self._batch_requests[idx % self._num_traces]


def create_req_generator(trace_dir, req_per_sec, arrival_type, random_seed, feature_dim):
    return RequestGenerator(trace_dir, req_per_sec, arrival_type, random_seed, feature_dim)
