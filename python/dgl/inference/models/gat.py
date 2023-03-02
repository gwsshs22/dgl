import time

import dgl
from dgl import DGLGraph
import dgl.function as fn

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.ops.edge_softmax import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils.internal import expand_as_pair

class DistGAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_outputs, num_layers, heads, activation=F.relu):
        super().__init__()
        assert len(heads) == num_layers, f"Number of heads (heads={heads}) should be the same with num_layers={num_layers}"
        self.activation = activation
        self.layers = nn.ModuleList()

        self.layers.append(GATConv(in_feats, n_hidden // heads[0], heads[0], feat_drop=0.6, attn_drop=0.6, allow_zero_in_degree=True, heads_aggregation='flatten', activation=activation))
        for i in range(num_layers - 2):
            self.layers.append(GATConv(n_hidden, n_hidden // heads[i + 1], heads[i + 1], feat_drop=0.6, attn_drop=0.6, allow_zero_in_degree=True, heads_aggregation='flatten', activation=activation))
        self.layers.append(GATConv(n_hidden, n_outputs, heads[-1], feat_drop=0.6, attn_drop=0.6, allow_zero_in_degree=True, heads_aggregation='mean', activation=None))

    def forward(self, blocks, h):
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)

        return h

# pylint: enable=W0235
class GATConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 heads_aggregation=None, # 'flatten' or 'mean
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()
        self.heads_aggregation = heads_aggregation
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

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

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            
            if self.heads_aggregation == "flatten":
                rst = rst.flatten(1)
            elif self.heads_aggregation == "mean":
                rst = rst.mean(1)

            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst

    # P3
    def p3_first_layer_mp(self, graph, feat):
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

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            return {
                "el": el,
                "er": er,
                "ft": feat_src
            }

    def p3_first_layer_dp(self, block, mp_aggr):
        with block.local_scope():
            block.srcdata.update({'ft': mp_aggr["ft"], 'el': mp_aggr["el"]})
            block.dstdata.update({'er': mp_aggr["er"]})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            block.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(block.edata.pop('e'))

            # compute softmax
            block.edata['a'] = self.attn_drop(edge_softmax(block, e))

            # message passing
            block.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = block.dstdata['ft']

            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * (rst.dim() - 2)), self._num_heads, self._out_feats)
            
            if self.heads_aggregation == "flatten":
                rst = rst.flatten(1)
            elif self.heads_aggregation == "mean":
                rst = rst.mean(1)

            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst

    def p3_split(self, start_idx, end_idx):
        fc_new_weight = nn.Parameter(self.fc.weight[:,start_idx:end_idx])

        self.fc = nn.Linear(fc_new_weight.shape[1], fc_new_weight.shape[0], bias=False)
        self.fc.weight = fc_new_weight

    # Vcut
    def div_names(self):
        return ["er"]

    def compute_dst_init_values(self, block, src_feats, num_local_dst_nodes):
        src_prefix_shape = src_feats.shape[:-1]
        feat_src = self.fc(src_feats).view(*src_prefix_shape, self._num_heads, self._out_feats)

        block.srcdata["feat_src"] = feat_src

        feat_dst = feat_src[:num_local_dst_nodes]
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        return { "er": er }

    def aggr_names(self):
        return ["logits_max", "exp_logits_sum", "ft"]

    def compute_aggregations(self, block, src_feats, dst_init_values):
        with block.local_scope():
            if "feat_src" in block.srcdata:
                feat_src = block.srcdata["feat_src"]
            else:
                src_prefix_shape = src_feats.shape[:-1]
                feat_src = self.fc(src_feats).view(*src_prefix_shape, self._num_heads, self._out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = dst_init_values["er"]

            block.srcdata.update({'ft': feat_src, 'el': el})
            block.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            block.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(block.edata.pop('e'))

            block.edata["logits"] = e
            block.update_all(fn.copy_e("logits", "h"), fn.max("h", "logits_max"))

            block.apply_edges(fn.e_sub_v("logits", "logits_max", "logits_recalculated"))
            block.edata["exp_logits"] = th.exp(block.edata.pop("logits_recalculated"))
            block.update_all(fn.copy_e("exp_logits", "h2"),
                             fn.sum("h2", "exp_logits_sum"))

            # compute softmax
            block.apply_edges(fn.e_div_v("exp_logits", "exp_logits_sum", "a"))
            
            # message passing
            block.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))

            return {
                "logits_max": block.dstdata["logits_max"],
                "exp_logits_sum": block.dstdata["exp_logits_sum"],
                "ft": block.dstdata["ft"]
            }

    def dmiv_names(self):
        return []

    def compute_dst_merge_init_values(self, dst_feats):
        return {}

    def merge(self, block, dst_merge_init_values, aggs_map):
        logits_max_aggs = aggs_map["logits_max"]
        exp_logits_sum_aggs = aggs_map["exp_logits_sum"]
        ft_aggs = aggs_map["ft"]

        logits_max_aggs[exp_logits_sum_aggs == 0] = -float('inf')
        logits_max, _ = logits_max_aggs.max(0)

        exp_max_logits_diff = th.exp(logits_max_aggs - logits_max)
        exp_logits_sum_aggs = exp_logits_sum_aggs * exp_max_logits_diff

        rst = ft_aggs * exp_logits_sum_aggs
        rst = rst.sum(0) / exp_logits_sum_aggs.sum(0)

        if self.bias is not None:
            rst = rst + self.bias.view(
                *((1,) * (rst.dim() - 2)), self._num_heads, self._out_feats)

        if self.heads_aggregation == "flatten":
            rst = rst.flatten(1)
        elif self.heads_aggregation == "mean":
            rst = rst.mean(1)

        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst
