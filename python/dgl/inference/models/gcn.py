import dgl
from dgl import DGLGraph
import dgl.function as fn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import dgl.function as fn
from dgl._ffi.base import DGLError
from dgl.utils.internal import expand_as_pair
from dgl.transforms.functional import reverse
from dgl.convert import block_to_graph
from dgl.heterograph import DGLBlock

class DistGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_outputs, n_layers, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, norm='right', allow_zero_in_degree=True, activation=activation))
        for i in range(1, n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, norm='right', allow_zero_in_degree=True, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_outputs, norm='right', allow_zero_in_degree=True, activation=None))

    # Normal forward pass
    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

        return h

# pylint: disable=W0235
class GraphConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            #
            # We only use 'right' for this experiment
            #
            # if self._norm in ['left', 'both']:
            #     degs = graph.out_degrees().float().clamp(min=1)
            #     if self._norm == 'both':
            #         norm = th.pow(degs, -0.5)
            #     else:
            #         norm = 1.0 / degs
            #     shp = norm.shape + (1,) * (feat_src.dim() - 1)
            #     norm = th.reshape(norm, shp)
            #     feat_src = feat_src * norm

            weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def compute_dst_init_values(self, block, src_feats, num_local_dst_nodes):
        return None

    def compute_aggregations(self, block, src_feats, num_local_dst_nodes, dst_init_values):
        with block.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')
            weight = self.weight
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                src_feats = th.matmul(src_feats, weight)
                block.srcdata['h'] = src_feats
                block.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = block.dstdata['h']
            else:
                # aggregate first then mult W
                block.srcdata['h'] = src_feats
                block.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = block.dstdata['h']
                rst = th.matmul(rst, weight)

            return {
                "sums": rst,
                "counts": block.in_degrees().float()
            }

    def merge(self, block, src_feats, num_local_dst_nodes, aggs_map):
        counts = aggs_map["counts"].sum(0).clamp(min=1)
        sums = aggs_map["sums"].sum(0)
        rst = sums / counts.reshape(-1, 1)

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
