"""Torch Module for Principal Neighbourhood Aggregation Convolution Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import torch
import torch.nn as nn

from .... import function as fn

def aggregate_mean(h):
    """mean aggregation"""
    return torch.mean(h, dim=1)


def aggregate_max(h):
    """max aggregation"""
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    """min aggregation"""
    return torch.min(h, dim=1)[0]


def aggregate_sum(h):
    """sum aggregation"""
    return torch.sum(h, dim=1)


def aggregate_std(h):
    """standard deviation aggregation"""
    return torch.sqrt(aggregate_var(h) + 1e-30)


def aggregate_var(h):
    """variance aggregation"""
    h_mean_squares = torch.mean(h * h, dim=1)
    h_mean = torch.mean(h, dim=1)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def _aggregate_moment(h, n):
    """moment aggregation: for each node (E[(X-E[X])^n])^{1/n}"""
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=1)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + 1e-30, 1.0 / n)
    return rooted_h_n


def aggregate_moment_3(h):
    """moment aggregation with n=3"""
    return _aggregate_moment(h, n=3)


def aggregate_moment_4(h):
    """moment aggregation with n=4"""
    return _aggregate_moment(h, n=4)


def aggregate_moment_5(h):
    """moment aggregation with n=5"""
    return _aggregate_moment(h, n=5)


def scale_identity(h):
    """identity scaling (no scaling operation)"""
    return h


def scale_amplification(h, D, delta):
    """amplification scaling"""
    return h * (np.log(D + 1) / delta)


def scale_attenuation(h, D, delta):
    """attenuation scaling"""
    return h * (delta / np.log(D + 1))


def torch_scale_amplification(h, D, delta):
    """amplification scaling"""
    return h * (torch.log(D + 1) / delta)


def torch_scale_attenuation(h, D, delta):
    """attenuation scaling"""
    return h * (delta / torch.log(D + 1))


AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": aggregate_moment_3,
    "moment4": aggregate_moment_4,
    "moment5": aggregate_moment_5,
}
SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}
TORCH_SCALERS = {
    "identity": scale_identity,
    "amplification": torch_scale_amplification,
    "attenuation": torch_scale_attenuation,
}

class PNAConvTower(nn.Module):
    """A single PNA tower in PNA layers"""

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        mem_optimized,
        dropout=0.0,
        edge_feat_size=0,
    ):
        super(PNAConvTower, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.mem_optimized = mem_optimized
        self.edge_feat_size = edge_feat_size

        if self.mem_optimized:
            self.M_src = nn.Linear(in_size, in_size)
            self.M_dst = nn.Linear(in_size, in_size)
        else:
            self.M = nn.Linear(2 * in_size + edge_feat_size, in_size)
        self.U = nn.Linear(
            (len(aggregators) * len(scalers) + 1) * in_size, out_size
        )
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_size)

    def reduce_func(self, nodes):
        """reduce function for PNA layer:
        tensordot of multiple aggregation and scaling operations"""
        msg = nodes.mailbox["msg"]
        degree = msg.size(1)
        h = torch.cat(
            [AGGREGATORS[agg](msg) for agg in self.aggregators], dim=1
        )
        h = torch.cat(
            [
                SCALERS[scaler](h, D=degree, delta=self.delta)
                if scaler != "identity"
                else h
                for scaler in self.scalers
            ],
            dim=1,
        )
        return {"h_neigh": h}

    def message(self, edges):
        """message function for PNA layer"""
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["a"]], dim=-1
            )
        else:
            f = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {"msg": self.M(f)}

    def forward(self, graph, node_feat, edge_feat=None):
        """compute the forward pass of a single tower in PNA convolution layer"""
        # calculate graph normalization factors
        # N = graph.num_dst_nodes()
        # snorm_n = torch.cat(
        #     [
        #         torch.ones(N, 1).to(node_feat) / N
        #     ],
        #     dim=0,
        # ).sqrt()

        with graph.local_scope():
            if self.mem_optimized:
                graph.srcdata["h_src"] = self.M_src(node_feat)
                dst_feat = node_feat[:graph.num_dst_nodes()]
                graph.dstdata['h_dst'] = self.M_dst(dst_feat)
                for aggr in self.aggregators:
                    if aggr == "min":
                        graph.update_all(fn.u_add_v("h_src", "h_dst", "min_msg"), fn.min("min_msg", "min"))
                    elif aggr == "max":
                        graph.update_all(fn.u_add_v("h_src", "h_dst", "max_msg"), fn.max("max_msg", "max"))
                    elif aggr == "mean":
                        graph.update_all(fn.u_add_v("h_src", "h_dst", "mean_msg"), fn.mean("mean_msg", "mean"))
                    else:
                        raise "Other types not yet supported for memory mem_optimized version."
                degree = graph.in_degrees().reshape(-1, 1)
                h_neigh = torch.cat(
                    [graph.dstdata.pop(agg) for agg in self.aggregators], dim=1
                )
                h_neigh = torch.cat(
                    [
                        TORCH_SCALERS[scaler](h_neigh, D=degree, delta=self.delta)
                        if scaler != "identity"
                        else h_neigh
                        for scaler in self.scalers
                    ],
                    dim=1,
                )
                graph.dstdata["h_neigh"] = h_neigh
            else:
                graph.srcdata["h"] = node_feat
                dst_feat = graph.srcdata['h'][:graph.num_dst_nodes()]
                graph.dstdata['h'] = dst_feat
                graph.update_all(self.message, self.reduce_func)
            h = self.U(torch.cat([dst_feat, graph.dstdata["h_neigh"]], dim=-1))
            # h = h * snorm_n
            return self.dropout(self.batchnorm(h))

        # snorm_n = torch.cat(
        #     [
        #         torch.ones(N, 1).to(node_feat) / N
        #         for N in graph.batch_num_nodes()
        #     ],
        #     dim=0,
        # ).sqrt()
        # with graph.local_scope():
        #     graph.ndata["h"] = node_feat
        #     if self.edge_feat_size > 0:
        #         assert edge_feat is not None, "Edge features must be provided."
        #         graph.edata["a"] = edge_feat

        #     graph.update_all(self.message, self.reduce_func)
        #     h = self.U(torch.cat([node_feat, graph.ndata["h_neigh"]], dim=-1))
        #     h = h * snorm_n
        #     return self.dropout(self.batchnorm(h))


class PNAConv(nn.Module):
    r"""Principal Neighbourhood Aggregation Layer from `Principal Neighbourhood Aggregation
    for Graph Nets <https://arxiv.org/abs/2004.05718>`__

    A PNA layer is composed of multiple PNA towers. Each tower takes as input a split of the
    input features, and computes the message passing as below.

    .. math::
        h_i^(l+1) = U(h_i^l, \oplus_{(i,j)\in E}M(h_i^l, e_{i,j}, h_j^l))

    where :math:`h_i` and :math:`e_{i,j}` are node features and edge features, respectively.
    :math:`M` and :math:`U` are MLPs, taking the concatenation of input for computing
    output features. :math:`\oplus` represents the combination of various aggregators
    and scalers. Aggregators aggregate messages from neighbours and scalers scale the
    aggregated messages in different ways. :math:`\oplus` concatenates the output features
    of each combination.

    The output of multiple towers are concatenated and fed into a linear mixing layer for the
    final output.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    aggregators : list of str
        List of aggregation function names(each aggregator specifies a way to aggregate
        messages from neighbours), selected from:

        * ``mean``: the mean of neighbour messages

        * ``max``: the maximum of neighbour messages

        * ``min``: the minimum of neighbour messages

        * ``std``: the standard deviation of neighbour messages

        * ``var``: the variance of neighbour messages

        * ``sum``: the sum of neighbour messages

        * ``moment3``, ``moment4``, ``moment5``: the normalized moments aggregation
        :math:`(E[(X-E[X])^n])^{1/n}`
    scalers: list of str
        List of scaler function names, selected from:

        * ``identity``: no scaling

        * ``amplification``: multiply the aggregated message by :math:`\log(d+1)/\delta`,
        where :math:`d` is the degree of the node.

        * ``attenuation``: multiply the aggregated message by :math:`\delta/\log(d+1)`
    delta: float
        The degree-related normalization factor computed over the training set, used by scalers
        for normalization. :math:`E[\log(d+1)]`, where :math:`d` is the degree for each node
        in the training set.
    dropout: float, optional
        The dropout ratio. Default: 0.0.
    num_towers: int, optional
        The number of towers used. Default: 1. Note that in_size and out_size must be divisible
        by num_towers.
    edge_feat_size: int, optional
        The edge feature size. Default: 0.
    residual : bool, optional
        The bool flag that determines whether to add a residual connection for the
        output. Default: True. If in_size and out_size of the PNA conv layer are not
        the same, this flag will be set as False forcibly.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import PNAConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = PNAConv(10, 10, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)
    >>> ret = conv(g, feat)
    """

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        num_towers=1,
        edge_feat_size=0,
        residual=True,
        mem_optimized=False
    ):
        super(PNAConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        assert (
            in_size % num_towers == 0
        ), "in_size must be divisible by num_towers"
        assert (
            out_size % num_towers == 0
        ), "out_size must be divisible by num_towers"
        self.tower_in_size = in_size // num_towers
        self.tower_out_size = out_size // num_towers
        self.edge_feat_size = edge_feat_size
        self.residual = residual
        if self.in_size != self.out_size:
            self.residual = False

        self.towers = nn.ModuleList(
            [
                PNAConvTower(
                    self.tower_in_size,
                    self.tower_out_size,
                    aggregators,
                    scalers,
                    delta,
                    dropout=dropout,
                    edge_feat_size=edge_feat_size,
                    mem_optimized=mem_optimized
                )
                for _ in range(num_towers)
            ]
        )

        self.mixing_layer = nn.Sequential(
            nn.Linear(out_size, out_size), nn.LeakyReLU()
        )

    def forward(self, graph, node_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute PNA layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            should be the same as out_size.
        """
        h_cat = torch.cat(
            [
                tower(
                    graph,
                    node_feat[
                        :,
                        ti * self.tower_in_size : (ti + 1) * self.tower_in_size,
                    ],
                    edge_feat,
                )
                for ti, tower in enumerate(self.towers)
            ],
            dim=1,
        )
        h_out = self.mixing_layer(h_cat)
        # add residual connection

        if graph.is_block:
            if self.residual:
                h_out = h_out + node_feat[:graph.num_dst_nodes()]
        else:
            if self.residual:
                h_out = h_out + node_feat

        return h_out
