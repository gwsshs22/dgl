"""Launching tool for DGL distributed training"""
import os
import stat
import sys
import subprocess
import argparse
import signal
import logging
import time
import json
import multiprocessing
import queue
import re
import random
from functools import partial
from threading import Thread
from typing import Optional

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from dgl.sampling import sample_neighbors as local_sample_neighbors
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[1], out_size, heads[2], residual=True, activation=None))
        
    def forward(self, blocks, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(blocks[i], h)
            if i == 2:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
        return h

def evaluate(blocks, features, labels, model):
    model.eval()
    with torch.no_grad():
        output = model(blocks, features)
        pred = np.where(output.data.cpu().numpy() >= 0, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), pred, average='micro')
        return score

def convert_to_block(g, seeds, features, labels, device, fanouts):
    blocks = []
    for fanout in fanouts:
        frontier = local_sample_neighbors(g, seeds.to(device), fanout)
        block = dgl.to_block(frontier).to(device)
        seeds = block.srcdata[dgl.NID]
        blocks.insert(0, block)
    return blocks, features[seeds].to(device), labels[seeds].to(device)

def main(args):
    device = 'cuda:0'
    in_size = 50
    out_size = 121
    model = GAT(in_size, 256, out_size, heads=[4,4,6]).to(device)
    model.load_state_dict(torch.load("/home/gwkim/gnn_models/gat_ppi_sampling.pt"))

    test_dataset = PPIDataset(mode='test')
    test_dataloader = GraphDataLoader(test_dataset, batch_size=2)

    total_score = 0
    for batch_id, batched_graph in enumerate(test_dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata['feat']
        labels = batched_graph.ndata['label']
        seeds = torch.arange(batched_graph.num_nodes())

        blocks, features, labels = convert_to_block(batched_graph, seeds, features, labels, device, [10, 20, 30])
        score = evaluate(blocks, features, labels, model)
        total_score += score
    f1_score = total_score / (batch_id + 1) # return average score
    print(f"f1_score={f1_score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)