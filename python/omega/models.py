from pathlib import Path
import json
import secrets

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from omega.utils import get_dataset_config

class GCN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, gcn_norm='right', activation=F.relu, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, allow_zero_in_degree=True, norm=gcn_norm))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True, norm=gcn_norm))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, allow_zero_in_degree=True, norm=gcn_norm))
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
    def __init__(self, in_size, hid_size, out_size, num_layers, heads, dropout, is_gatv2):
        super().__init__()
        assert(num_layers == len(heads))

        def create_layer(
            in_feats,
            out_feats,
            num_heads,
            res,
            act):
            if is_gatv2:
                return dglnn.GATv2ConvOrg(
                    in_feats,
                    out_feats,
                    num_heads,
                    residual=res,
                    activation=act,
                    allow_zero_in_degree=True)
            else:
                return dglnn.GATConv(
                    in_feats,
                    out_feats,
                    num_heads,
                    residual=res,
                    activation=act,
                    allow_zero_in_degree=True)

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(create_layer(in_size, hid_size // heads[0], heads[0], res=False, act=F.elu))
        for i in range(num_layers - 2):
            self.gat_layers.append(create_layer(hid_size, hid_size // heads[i + 1], heads[i + 1], res=True, act=F.elu))
        self.gat_layers.append(create_layer(hid_size, out_size, heads[-1], res=True, act=None))
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

def create_model(gnn, num_inputs, num_hiddens, num_classes, num_layers, gat_heads, gcn_norm='right', dropout=0.0):
    if gnn == "gcn":
        model = GCN(num_inputs, num_hiddens, num_classes, num_layers, gcn_norm=gcn_norm, dropout=dropout)
    elif gnn == "sage":
        model = SAGE(num_inputs, num_hiddens, num_classes, num_layers, dropout=dropout, aggr='mean')
    elif gnn == "gat" or gnn == "gatv2":
        is_gatv2 = gnn == "gatv2"
        model = GAT(num_inputs, num_hiddens, num_classes, num_layers, heads=gat_heads, dropout=dropout, is_gatv2=is_gatv2)
    return model

def load_training_config(training_config_path):
    training_config = json.loads(training_config_path.read_text())

    if "id" not in training_config:
        training_config["id"] = secrets.token_hex(10)
        with open(training_config_path, "w") as f:
            f.write(json.dumps(training_config, indent=4, sort_keys=True))
            f.write("\n")
    
    if "gcn_norm" not in training_config:
        training_config["gcn_norm"] = "both"
        with open(training_config_path, "w") as f:
            f.write(json.dumps(training_config, indent=4, sort_keys=True))
            f.write("\n")
    
    return training_config

def load_model_from(training_dir, for_omega=False):
    training_dir = Path(training_dir)
    model_path = training_dir / "model.pt"
    config_path = training_dir / "config.json"
    assert model_path.exists()
    assert config_path.exists()

    training_config = load_training_config(config_path)
    dataset_config = get_dataset_config(training_config["graph_name"])

    gnn = training_config["gnn"]
    num_layers = training_config["num_layers"]
    gat_heads = [int(h) for h in training_config["gat_heads"].split(",")]
    if gnn == "gat" or gnn == "gatv2":
        assert all([h > 0 for h in gat_heads])
        assert len(gat_heads) == num_layers

    model = create_model(
        training_config["gnn"],
        dataset_config.num_inputs,
        training_config["num_hiddens"],
        dataset_config.num_classes,
        num_layers,
        gat_heads)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model, training_config, dataset_config
