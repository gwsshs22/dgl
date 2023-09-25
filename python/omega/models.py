from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from omega.utils import get_dataset_config

class GCN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def layer_foward(self, layer_idx, block, inputs):
        h = self.layers[layer_idx](block, inputs)
        if layer_idx != len(self.layers) - 1:
            h = self.activation(h)
            h = self.dropout(h)
        return h


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.5, aggr="mean"):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggr))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggr))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggr))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def layer_foward(self, layer_idx, block, inputs):
        h = self.layers[layer_idx](block, inputs)
        if layer_idx != len(self.layers) - 1:
            h = self.activation(h)
            h = self.dropout(h)
        return h

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, heads, dropout):
        super().__init__()
        assert(num_layers == len(heads))
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(dglnn.GATConv(in_size, hid_size // heads[0], heads[0], activation=F.elu, allow_zero_in_degree=True))
        for i in range(num_layers - 2):
            self.gat_layers.append(dglnn.GATConv(hid_size, hid_size // heads[i + 1], heads[i + 1], residual=True, activation=F.elu, allow_zero_in_degree=True))
        self.gat_layers.append(dglnn.GATConv(hid_size, out_size, heads[-1], residual=True, activation=None, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(blocks[i], h)
            if i == self.num_layers - 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
                h = self.dropout(h)
        return h
    
    def layer_foward(self, layer_idx, block, inputs):
        h = self.gat_layers[layer_idx](block, inputs)
        if layer_idx == self.num_layers - 1:  # last layer 
            h = h.mean(1)
        else:       # other layer(s)
            h = h.flatten(1)
            h = self.dropout(h)
        return h

def create_model(gnn, num_inputs, num_hiddens, num_classes, num_layers, gat_heads, dropout=0.0):
    if gnn == "gcn":
        model = GCN(num_inputs, num_hiddens, num_classes, num_layers, dropout=dropout)
    elif gnn == "sage":
        model = SAGE(num_inputs, num_hiddens, num_classes, num_layers, dropout=dropout, aggr='mean')
    elif gnn == "gat":
        model = GAT(num_inputs, num_hiddens, num_classes, num_layers, heads=gat_heads, dropout=dropout)
    return model

def load_model_from(training_dir):
    training_dir = Path(training_dir)
    model_path = training_dir / "model.pt"
    config_path = training_dir / "config.json"
    assert model_path.exists()
    assert config_path.exists()

    training_config = json.loads(config_path.read_text())
    dataset_config = get_dataset_config(training_config["graph_name"])

    gnn = training_config["gnn"]
    num_layers = training_config["num_layers"]
    gat_heads = [int(h) for h in training_config["gat_heads"].split(",")]
    if gnn == "gat":
        assert all([h > 0 for h in gat_heads])
        assert len(gat_heads) == num_layers

    model = create_model(
        training_config["gnn"],
        dataset_config.num_inputs,
        training_config["num_hiddens"],
        dataset_config.num_classes,
        num_layers,
        gat_heads)
    
    model.load_state_dict(torch.load(model_path))
    return model, training_config, dataset_config
