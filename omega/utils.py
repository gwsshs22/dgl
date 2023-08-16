import time
import contextlib
import os
from functools import namedtuple
import json

import torch

import dgl
from dgl.data import RedditDataset

from ogb.nodeproppred import DglNodePropPredDataset

import numpy as np
import scipy.sparse

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def load_graph(graph_name):
    
    if graph_name == "reddit":
        return RedditDataset(self_loop=True)[0]
    elif graph_name == "ogb-product":
        return DglNodePropPredDataset(name="ogbn-products")[0][0]
    else:
        # TODO(gwkim): add other datasets
        raise f"{graph_name} is not supported yet."

class Timer:
    def __init__(self):
        self.values = {}

    @contextlib.contextmanager
    def measure(self, key, cuda_sync=False, device=None):
        try:
            start_time = time.time()
            yield
        finally:
            if cuda_sync:
                torch.cuda.synchronize(device)

            elapsed = time.time() - start_time
            self.add(key, elapsed)

    def add(self, key, value):
        if key in self.values:
            self.values[key] += value
        else:
            self.values[key] = value

    def clear(self):
        self.values = {}
    
    def get_values(self):
        return self.values
