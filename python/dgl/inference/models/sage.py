import time

import dgl
from dgl import DGLGraph
import dgl.function as fn

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl._ffi.base import DGLError
from dgl.utils.internal import expand_as_pair, check_eq_shape
from dgl.base import dgl_warning

class DistSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean', activation=activation))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean', activation=activation))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean', activation=None))
        self.activation = activation

    # Normal forward pass
    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

        return h

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {'mean', 'gcn', 'pool', 'lstm'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(th.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)


    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_src('h', 'm')
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)

            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # We don't use GraphSAGE GCN.
            h_self = self.fc_self(h_self)
            rst = h_self + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)

            return rst

    # P3
    def p3_first_layer_mp(self, graph, feat):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_src('h', 'm')
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)

            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # We don't use GraphSAGE GCN.
            h_self = self.fc_self(h_self)
            rst = h_self + h_neigh

            return {
                "rst": rst
            }

    def p3_first_layer_dp(self, block, mp_aggr):
        rst = mp_aggr["rst"]
        # bias term
        if self.bias is not None:
            rst = rst + self.bias

        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
            
        return rst

    def p3_split(self, start_idx, end_idx):
        fc_self_new_weight = nn.Parameter(self.fc_self.weight[:,start_idx:end_idx])
        fc_neigh_new_weight = nn.Parameter(self.fc_neigh.weight[:,start_idx:end_idx])

        self.fc_self = nn.Linear(fc_self_new_weight.shape[1], fc_self_new_weight.shape[0], bias=False)
        self.fc_self.weight = fc_self_new_weight
        self.fc_neigh = nn.Linear(fc_neigh_new_weight.shape[1], fc_neigh_new_weight.shape[0], bias=False)
        self.fc_neigh.weight = fc_neigh_new_weight

    # VCUT
    def div_names(self):
        return []

    def compute_dst_init_values(self, block, src_feats, num_local_dst_nodes):
        return {}

    def aggr_names(self):
        return ["neigh_counts", "neigh_sums"]

    def compute_aggregations(self, block, src_feats, dst_init_values):
        with block.local_scope():
            msg_fn = fn.copy_src('h', 'm')

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                block.srcdata['h'] = self.fc_neigh(src_feats) if lin_before_mp else src_feats
                block.update_all(msg_fn, fn.sum('m', 'neigh_sum'))
                h_neigh_sum = block.dstdata['neigh_sum']
                if not lin_before_mp:
                    h_neigh_sum = self.fc_neigh(h_neigh_sum)
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            return {
                "neigh_counts": block.in_degrees().float(),
                "neigh_sums": h_neigh_sum
            }

    def dmiv_names(self):
        return ["h_self"]

    def compute_dst_merge_init_values(self, dst_feats):
        return {
            "h_self": self.fc_self(dst_feats)
        }

    def merge(self, block, dst_merge_init_values, aggs_map):
        h_neigh_counts = aggs_map["neigh_counts"].sum(0).clamp(min=1)
        h_neigh_sums = aggs_map["neigh_sums"].sum(0)
        h_neigh = h_neigh_sums / h_neigh_counts.reshape(-1, 1)

        h_self = dst_merge_init_values["h_self"]
        rst = h_self + h_neigh

        # bias term
        if self.bias is not None:
            rst = rst + self.bias

        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)

        return rst
