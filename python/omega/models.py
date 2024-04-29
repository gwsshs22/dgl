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
        self, in_feats, n_hidden, n_classes, n_layers, gcn_norm='both', activation=F.relu, dropout=0.5):
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

    def feature_preprocess(self, x):
        return x

    def forward(self, blocks, x, saint_normalize=False):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            if saint_normalize:
                h = layer(block, h, edge_weight=block.edata["a_n"])
            else:
                h = layer(block, h)
            
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def layer_foward(self, layer_idx, block, inputs, h0=None):
        h = self.layers[layer_idx](block, inputs)
        if layer_idx != len(self.layers) - 1:
            h = self.activation(h)
            h = self.dropout(h)
        return h


class GCN2(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, gcn_norm='both', activation=F.relu, alpha=0.5, lamb=1.0):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.fc1 = nn.Linear(in_feats, n_hidden)
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GCN2Conv(n_hidden, 1, alpha=alpha, lambda_=lamb, allow_zero_in_degree=True, norm=gcn_norm))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GCN2Conv(n_hidden, i + 1, alpha=alpha, lambda_=lamb, allow_zero_in_degree=True, norm=gcn_norm))
        self.layers.append(dglnn.GCN2Conv(n_hidden, n_layers, alpha=alpha, lambda_=lamb, allow_zero_in_degree=True, norm=gcn_norm))
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(0.2)
        self.activation = activation

    def reset_parameters(self):
        nn.init.normal_(self.fc1)
        nn.init.normal_(self.fc2)
        for l in self.layers:
            l.reset_parameters()

    def feature_preprocess(self, x):
        x = self.dropout(x)
        return self.activation(self.fc1(x))

    def forward(self, blocks, x, saint_normalize=False):
        h0 = self.feature_preprocess(x)

        h = h0
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            identity = h
            h = layer(block, h, h0, saint_normalize=saint_normalize)
            h += identity[:h.shape[0]]
            h = self.activation(h)
            h = self.dropout(h)

        return self.fc2(h)

    def layer_foward(self, layer_idx, block, inputs, h0):
        h = self.layers[layer_idx](block, inputs, h0)
        h += inputs[:h.shape[0]]
        h = self.activation(h)
        h = self.dropout(h)

        if layer_idx == self.n_layers - 1:
            h = self.fc2(h)

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

    def feature_preprocess(self, x):
        return x

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def layer_foward(self, layer_idx, block, inputs, h0=None):
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

    def feature_preprocess(self, x):
        return x

    def forward(self, blocks, inputs, get_attention=False):
        attns = []
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            if get_attention:
                h, attn = layer(blocks[i], h, get_attention=True)
                attns.append(attn)
            else:
                h = layer(blocks[i], h)

            if i == self.num_layers - 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
                h = self.dropout(h)

        if get_attention:
            return h, attns
        else:
            return h

    def layer_foward(self, layer_idx, block, inputs, h0=None, get_attention=False):
        if get_attention:
            h, attn = self.gat_layers[layer_idx](block, inputs, get_attention=True)
        else:
            h = self.gat_layers[layer_idx](block, inputs)
        if layer_idx == self.num_layers - 1:  # last layer 
            h = h.mean(1)
        else:       # other layer(s)
            h = h.flatten(1)
            h = self.dropout(h)
        
        if get_attention:
            return h, attn
        else:
            return h

class GINMLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class GIN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.5, aggr="mean"):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer_idx in range(n_layers):
            if layer_idx == 0:
                mlp = GINMLP(in_feats, n_hidden, n_hidden)
                self.batch_norms.append(nn.BatchNorm1d(n_hidden))
            elif layer_idx < n_layers - 1:
                mlp = GINMLP(n_hidden, n_hidden, n_hidden)
                self.batch_norms.append(nn.BatchNorm1d(n_hidden))
            else:
                mlp = GINMLP(n_hidden, n_hidden, n_classes)

            self.layers.append(
                dglnn.GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def feature_preprocess(self, x):
        return x

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.batch_norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def layer_foward(self, layer_idx, block, inputs, h0=None):
        h = self.layers[layer_idx](block, inputs)
        if layer_idx != len(self.layers) - 1:
            h = self.batch_norms[layer_idx](h)
            h = self.activation(h)
            h = self.dropout(h)
        return h


class PNA(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, delta, activation=F.relu, dropout=0.0,
        aggregators=['mean', 'min', 'max'], scalers=['identity', 'amplification', 'attenuation'],
        mem_optimized=True):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        self.layers.append(dglnn.PNAConv(in_feats, n_hidden, aggregators, scalers, delta, mem_optimized=mem_optimized))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.PNAConv(n_hidden, n_hidden, aggregators, scalers, delta, mem_optimized=mem_optimized))
        self.layers.append(dglnn.PNAConv(n_hidden, n_classes, aggregators, scalers, delta, mem_optimized=mem_optimized))
        self.activation = activation

    def feature_preprocess(self, x):
        return x

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
        return h

    def layer_foward(self, layer_idx, block, inputs, h0=None):
        h = self.layers[layer_idx](block, inputs)
        if layer_idx != len(self.layers) - 1:
            h = self.activation(h)
        return h

def create_model(gnn, num_inputs, num_hiddens, num_classes, num_layers, gat_heads, gcn_norm='both', gcn2_alpha=0.5, dropout=0.0, pna_delta=-1.0):
    if gnn == "gcn":
        model = GCN(num_inputs, num_hiddens, num_classes, num_layers, gcn_norm=gcn_norm, dropout=dropout)
    elif gnn == "gcn2":
        model = GCN2(num_inputs, num_hiddens, num_classes, num_layers, gcn_norm=gcn_norm, alpha=gcn2_alpha)
    elif gnn == "sage":
        model = SAGE(num_inputs, num_hiddens, num_classes, num_layers, dropout=dropout, aggr='mean')
    elif gnn == "gat" or gnn == "gatv2":
        is_gatv2 = gnn == "gatv2"
        model = GAT(num_inputs, num_hiddens, num_classes, num_layers, heads=gat_heads, dropout=dropout, is_gatv2=is_gatv2)
    elif gnn == "gin":
        model = GIN(num_inputs, num_hiddens, num_classes, num_layers, dropout=dropout)
    elif gnn == "pna":
        assert pna_delta > 0.0
        model = PNA(num_inputs, num_hiddens, num_classes, num_layers, pna_delta, mem_optimized=True)
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
    
    if gnn == "gat" or gnn == "gatv2":
        gat_heads = [int(h) for h in training_config["gat_heads"].split(",")]
        assert all([h > 0 for h in gat_heads])
        assert len(gat_heads) == num_layers
    else:
        gat_heads = [8] * num_layers

    if gnn == "pna":
        pna_delta = training_config["pna_delta"]
    else:
        pna_delta = -1.0

    if gnn == "gcn2":
        assert "gcn2_alpha" in training_config
        gcn2_alpha = training_config["gcn2_alpha"]
    else:
        gcn2_alpha = 0.5

    model = create_model(
        training_config["gnn"],
        dataset_config.num_inputs,
        training_config["num_hiddens"],
        dataset_config.num_classes,
        num_layers,
        gat_heads,
        pna_delta=pna_delta,
        gcn_norm=training_config['gcn_norm'],
        gcn2_alpha=gcn2_alpha)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model, training_config, dataset_config
