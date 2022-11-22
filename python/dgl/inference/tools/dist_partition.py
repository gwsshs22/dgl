import json
import os
import time
import numpy as np
import gc

from dgl import backend as F
from dgl import utils
from dgl.heterograph import DGLHeteroGraph
from dgl.convert import to_homogeneous
from dgl.partition import reshuffle_graph, get_peak_mem, c_api_dgl_partition_with_halo_hetero, _get_halo_heterosubgraph_inner_node
from dgl.base import NID, EID, NTYPE, ETYPE, dgl_warning
from dgl.random import choice as random_choice
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors

def _get_inner_node_mask(graph, ntype_id):
    if NTYPE in graph.ndata:
        dtype = F.dtype(graph.ndata['inner_node'])
        return graph.ndata['inner_node'] * F.astype(graph.ndata[NTYPE] == ntype_id, dtype) == 1
    else:
        return graph.ndata['inner_node'] == 1

def partition_graph_with_halo(g, node_part, extra_cached_hops, reshuffle=False):
    assert len(node_part) == g.number_of_nodes()
    if reshuffle:
        g, node_part = reshuffle_graph(g, node_part)
        orig_nids = g.ndata['orig_id']
        orig_eids = g.edata['orig_id']

    node_part = utils.toindex(node_part)
    start = time.time()
    subgs = c_api_dgl_partition_with_halo_hetero(
        g._graph, node_part.todgltensor(), extra_cached_hops)
    # g is no longer needed. Free memory.
    g = None
    print('Split the graph: {:.3f} seconds'.format(time.time() - start))
    subg_dict = {}
    node_part = node_part.tousertensor()
    start = time.time()

    # This creaets a subgraph from subgraphs returned from the CAPI above.
    def create_subgraph(subg, induced_nodes, induced_edges, inner_node):
        subg1 = DGLHeteroGraph(gidx=subg.graph, ntypes=['_N'], etypes=['_E'])
        subg1.ndata[NID] = induced_nodes[0]
        return subg1

    for i, subg in enumerate(subgs):
        inner_node = _get_halo_heterosubgraph_inner_node(subg)
        inner_node = F.zerocopy_from_dlpack(inner_node.to_dlpack())

        subg = create_subgraph(subg, subg.induced_nodes, subg.induced_edges, inner_node)
        subg.ndata['inner_node'] = inner_node
        subg.ndata['part_id'] = F.gather_row(node_part, subg.ndata[NID])
        if reshuffle:
            subg.ndata['orig_id'] = F.gather_row(orig_nids, subg.ndata[NID])

        subg_dict[i] = subg
    print('Construct subgraphs: {:.3f} seconds'.format(time.time() - start))
    if reshuffle:
        return subg_dict, orig_nids, orig_eids
    else:
        return subg_dict, None, None

def get_homogeneous(g, balance_ntypes):
    if g.is_homogeneous:
        sim_g = to_homogeneous(g)
        if isinstance(balance_ntypes, dict):
            assert len(balance_ntypes) == 1
            bal_ntypes = list(balance_ntypes.values())[0]
        else:
            bal_ntypes = balance_ntypes
    elif isinstance(balance_ntypes, dict):
        # Here we assign node types for load balancing.
        # The new node types includes the ones provided by users.
        num_ntypes = 0
        for key in g.ntypes:
            if key in balance_ntypes:
                g.nodes[key].data['bal_ntype'] = F.astype(balance_ntypes[key],
                                                          F.int32) + num_ntypes
                uniq_ntypes = F.unique(balance_ntypes[key])
                assert np.all(F.asnumpy(uniq_ntypes) == np.arange(len(uniq_ntypes)))
                num_ntypes += len(uniq_ntypes)
            else:
                g.nodes[key].data['bal_ntype'] = F.ones((g.number_of_nodes(key),), F.int32,
                                                        F.cpu()) * num_ntypes
                num_ntypes += 1
        sim_g = to_homogeneous(g, ndata=['bal_ntype'])
        bal_ntypes = sim_g.ndata['bal_ntype']
        print('The graph has {} node types and balance among {} types'.format(
            len(g.ntypes), len(F.unique(bal_ntypes))))
        # We now no longer need them.
        for key in g.ntypes:
            del g.nodes[key].data['bal_ntype']
        del sim_g.ndata['bal_ntype']
    else:
        sim_g = to_homogeneous(g)
        bal_ntypes = sim_g.ndata[NTYPE]
    return sim_g, bal_ntypes

def partition_graph(g, graph_name, num_parts, out_path, num_hops=1, part_method="metis",
                    reshuffle=True, balance_ntypes=None, balance_edges=False, return_mapping=False,
                    num_trainers_per_machine=1, objtype='cut'):

    if objtype not in ['cut', 'vol']:
        raise ValueError

    if not reshuffle:
        dgl_warning("The argument reshuffle will be deprecated in the next release. "
                    "For heterogeneous graphs, reshuffle must be enabled.")
        exit(-1)

    assert num_parts > 1 and part_method == "random", "Only support random partition + multiple partition"

    start = time.time()
    sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
    print('Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB'.format(
        time.time() - start, get_peak_mem()))

    ntypes = {ntype:g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype:g.get_etype_id(etype) for etype in g.etypes}
    number_of_nodes = g.number_of_nodes()
    number_of_edges = g.number_of_edges()

    del g
    gc.collect()

    node_parts = random_choice(num_parts, sim_g.number_of_nodes())
    start = time.time()
    parts, _, _ = partition_graph_with_halo(sim_g, node_parts, num_hops, reshuffle=reshuffle)
    print('Splitting the graph into partitions takes {:.3f}s, peak mem: {:.3f} GB'.format(
        time.time() - start, get_peak_mem()))

    os.makedirs(out_path, mode=0o775, exist_ok=True)
    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)

    # With reshuffling, we can ensure that all nodes and edges are reshuffled
    # and are in contiguous ID space.
    node_map_val = {}
    edge_map_val = {}
    for ntype, ntype_id in ntypes.items():
        val = []
        node_map_val[ntype] = []
        for i in parts:
            inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
            val.append(F.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0)))
            inner_nids = F.boolean_mask(parts[i].ndata[NID], inner_node_mask)
            node_map_val[ntype].append([int(F.as_scalar(inner_nids[0])),
                                        int(F.as_scalar(inner_nids[-1])) + 1])
        val = np.cumsum(val).tolist()
        # assert val[-1] == g.number_of_nodes(ntype)

    # Double check that the node IDs in the global ID space are sorted.
    for ntype in node_map_val:
        val = np.concatenate([np.array(l) for l in node_map_val[ntype]])
        assert np.all(val[:-1] <= val[1:])
    for etype in edge_map_val:
        val = np.concatenate([np.array(l) for l in edge_map_val[etype]])
        assert np.all(val[:-1] <= val[1:])

    start = time.time()
    part_metadata = {'graph_name': graph_name,
                     'num_nodes': number_of_nodes,
                     'num_edges': number_of_edges,
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val,
                     'ntypes': ntypes,
                     'etypes': etypes}

    for part_id in range(num_parts):
        part = parts[part_id]

        # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}

        # for ntype, ntype_id in ntypes.items():
        #     # To get the edges in the input graph, we should use original node IDs.
        #     # Both orig_id and NID stores the per-node-type IDs.
        #     ndata_name = 'orig_id' if reshuffle else NID
        #     inner_node_mask = _get_inner_node_mask(part, ntype_id)
        #     # This is global node IDs.
        #     local_nodes = F.boolean_mask(part.ndata[ndata_name], inner_node_mask)
        #     print('part {} has {} nodes and {} are inside the partition'.format(
        #         part_id, part.number_of_nodes(), len(local_nodes)))

        #     for name in g.nodes[ntype].data:
        #         if name in [NID, 'inner_node']:
        #             continue
        #         node_feats[ntype + '/' + name] = F.gather_row(g.nodes[ntype].data[name],
        #                                                       local_nodes)

        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {
            'node_feats': os.path.relpath(node_feat_file, out_path),
            'edge_feats': os.path.relpath(edge_feat_file, out_path),
            'part_graph': os.path.relpath(part_graph_file, out_path)}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)

        save_graphs(part_graph_file, [part])

    print('Save partitions: {:.3f} seconds, peak memory: {:.3f} GB'.format(
        time.time() - start, get_peak_mem()))

    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

    print("Done partitioning")
