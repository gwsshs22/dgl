import time
import contextlib
from dataclasses import dataclass
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

@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    num_inputs: int
    multilabel: bool
    inductive: bool

dataset_configs = {
    "reddit": DatasetConfig("reddit", 41, 602, False, True),
    "ogbn-products": DatasetConfig("ogbn-products", 47, 100, False, False),
    "ogbn-papers100M": DatasetConfig("ogbn-papers100M", 172, 128, False, False),
    "flickr": DatasetConfig("flickr", 7, 500, False, True),
    "yelp": DatasetConfig("yelp", 100, 300, True, True),
    "amazon": DatasetConfig("amazon", 107, 200, True, True),
    "fb5b": DatasetConfig("fb5b", 128, 16, False, False),
    "fb10b": DatasetConfig("fb10b", 128, 16, False, False),
}

def get_dataset_config(graph_name):
    return dataset_configs[graph_name]

def load_graph(graph_name, ogbn_data_root=None, saint_data_root=None):
    if graph_name == "reddit":
        g = RedditDataset(self_loop=True)[0]
        g.ndata["labels"] = g.ndata.pop("label")
        g.ndata["features"] = g.ndata.pop("feat")
        g.edata.pop("__orig__")
        return  g
    elif graph_name == "ogb-product" or graph_name == "ogbn-products":
        return load_ogbs(ogbn_data_root, "ogbn-products")
    elif graph_name == "ogb-papers100M" or graph_name == "ogbn-papers100M":
        return load_ogbs(ogbn_data_root, "ogbn-papers100M")
    elif graph_name == "amazon" or graph_name == "flickr" or graph_name == "yelp":
        multilabel = graph_name != "flickr"
        return load_saint(saint_data_root, graph_name, multilabel)
    else:
        # TODO(gwkim): add other datasets
        raise f"{graph_name} is not supported yet."

def load_ogbs(ogbn_data_root, graph_name):
    data = DglNodePropPredDataset(name=graph_name, root=ogbn_data_root)
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    labels = labels[:, 0]

    g.ndata["labels"] = labels
    g.ndata["features"] = g.ndata.pop("feat")
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    train_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["test_mask"] = test_mask
    return g

def load_saint(saint_data_root, graph_name, multilabel):
    prefix = f"{saint_data_root}/{graph_name}"
    dgl_graph_path = f"{prefix}/{graph_name}.dgl"
    if os.path.exists(dgl_graph_path):
        return dgl.load_graphs(dgl_graph_path)[0][0]

    adj_full = scipy.sparse.load_npz("{}/adj_full.npz".format(prefix)).astype(
        bool
    )
    g = dgl.from_scipy(adj_full)
    num_nodes = g.num_nodes()

    adj_train = scipy.sparse.load_npz(
        "{}/adj_train.npz".format(prefix)
    ).astype(bool)
    train_nid = np.array(list(set(adj_train.nonzero()[0])))

    role = json.load(open("{}/role.json".format(prefix)))
    mask = np.zeros((num_nodes,), dtype=bool)
    train_mask = mask.copy()
    train_mask[role["tr"]] = True
    val_mask = mask.copy()
    val_mask[role["va"]] = True
    test_mask = mask.copy()
    test_mask[role["te"]] = True

    feats = np.load("{}/feats.npy".format(prefix))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    class_map = json.load(open("{}/class_map.json".format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}

    if multilabel:
        # Multi-label binary classification
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_nodes, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_nodes,))
        for k, v in class_map.items():
            class_arr[k] = v

    g.ndata["features"] = torch.tensor(feats, dtype=torch.float)
    g.ndata["labels"] = torch.tensor(
        class_arr, dtype=torch.float if multilabel else torch.long
    )
    g.ndata["train_mask"] = torch.tensor(train_mask, dtype=torch.bool)
    g.ndata["val_mask"] = torch.tensor(val_mask, dtype=torch.bool)
    g.ndata["test_mask"] = torch.tensor(test_mask, dtype=torch.bool)

    dgl.save_graphs(dgl_graph_path, [g])
    return g

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
