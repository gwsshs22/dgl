import argparse
import os

import dgl
import numpy as np
import torch
import json

def find_index_of_values(arr, sorter, values):
    return sorter[np.searchsorted(arr, values, sorter=sorter)]

def build_id_mappings(output_dir, num_parts, part_data_list):
    orig_ids = []
    cumsum_nodes = 0
    num_nodes_in_partitions = []
    cumsum_nodes_list = []
    id_ranges = []
    for i in range(num_parts):
        num_nodes_in_partition = part_data_list[i]['inner_node'].sum().tolist()
        num_nodes_in_partitions.append(num_nodes_in_partition)
        orig_ids.append(part_data_list[i]["orig_id"][:num_nodes_in_partition])
        id_ranges.append([cumsum_nodes, cumsum_nodes + num_nodes_in_partition])
        cumsum_nodes_list.append(cumsum_nodes)
        cumsum_nodes += num_nodes_in_partition

    orig_id_to_global_ids = []

    for i in range(num_parts):
        orig_id_to_global_ids.append(part_data_list[i]['orig_id'][:num_nodes_in_partitions[i]])
    
    orig_id_to_global_id = torch.concat(orig_id_to_global_ids)
    sorter = np.argsort(orig_id_to_global_id)

    orig_ids_in_partitions = torch.concat(orig_ids)

    dgl.data.save_tensors(f"{output_dir}/id_mappings.dgl", {
        "orig_ids_in_partitions": orig_ids_in_partitions
    })

    new_part_data_list = []
    for i in range(num_parts):
        new_part_data_list.append({
            "orig_id": part_data_list[i]["orig_id"],
            dgl.NID: find_index_of_values(orig_id_to_global_id, sorter, part_data_list[i]['orig_id']),
            "part_id": part_data_list[i]["part_id"],
            "inner_node": part_data_list[i]["inner_node"]
        })

    return new_part_data_list, id_ranges


def save_infer_target_graph(input_dir, output_dir):
    infer_data = dgl.data.load_tensors(input_dir + "/infer_data.dgl")
    print(f"num_infer_targets = {infer_data['infer_target_mask'].sum()}")
    
    infer_g = dgl.graph((infer_data["new_u"], infer_data["new_v"]))
    infer_g.ndata[dgl.NID] = infer_data[dgl.NID]
    infer_g.ndata["infer_target_mask"] = infer_data["infer_target_mask"]
    dgl.save_graphs(f"{output_dir}/infer_target_graph.dgl", [infer_g])

def save_part_graph(input_dir, output_dir, part_id, cumsum_edge, new_part_data):
    edges_data = dgl.data.load_tensors(input_dir + f"/part{part_id}/edges.dgl")
    u, v = edges_data["new_u"], edges_data["new_v"]
    num_edges = u.shape[0]

    part_g = dgl.graph((u, v))
    part_g.ndata[dgl.NID] = new_part_data[dgl.NID]
    part_g.ndata["orig_id"] = new_part_data["orig_id"]
    part_g.ndata["inner_node"] = new_part_data["inner_node"]
    part_g.ndata["part_id"] = new_part_data["part_id"]
    part_g.edata[dgl.NID] = torch.arange(cumsum_edge, cumsum_edge + num_edges)

    dgl.save_graphs(output_dir + f"/part{part_id}/graph.dgl", [part_g])
    return num_edges

def save_graphs(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_parts = args.num_parts
    graph_name = args.graph_name

    os.makedirs(output_dir, exist_ok=True)

    save_infer_target_graph(input_dir, output_dir)

    part_data_list = []
    for i in range(num_parts):
        part_data_list.append(dgl.data.load_tensors(input_dir + f"/part{i}/data.dgl"))

    new_part_data_list, id_ranges = build_id_mappings(output_dir, num_parts, part_data_list)

    cumsum_edge = 0
    edge_ranges = []
    for i in range(num_parts):
        num_edge = save_part_graph(input_dir, output_dir, i, cumsum_edge, new_part_data_list[i])
        edge_ranges.append([cumsum_edge, cumsum_edge + num_edge])
        cumsum_edge += num_edge

    config = {
        "edge_map": {},
        "etypes": { "_E": 0 },
        "graph_name": graph_name,
        "halo_hops": 1,
        "node_map": {
            "_N": id_ranges
        },
        "ntypes": { "_N": 0 },
        "num_edges": cumsum_edge,
        "num_nodes": id_ranges[-1][-1],
        "num_parts": num_parts
    }

    for i in range(num_parts):
        config[f"part-{i}"] = {
            "edge_feats": f"part{i}/edge_feat.dgl",
            "node_feats": f"part{i}/node_feat.dgl",
            "part_graph": f"part{i}/graph.dgl",
        }

    config["part_method"] = 'random'

    with open(f"{output_dir}/{graph_name}.json", "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)


def fill_infer_targets_features(output_dir, num_features):
    infer_g = dgl.load_graphs(f"{output_dir}/infer_target_graph.dgl")[0][0]
    num_nodes_in_infer_g = infer_g.number_of_nodes()
    id_mappings = dgl.data.load_tensors(f"{output_dir}/id_mappings.dgl")
    id_mappings["infer_target_features"] = torch.rand(num_nodes_in_infer_g, num_features)
    dgl.data.save_tensors(f"{output_dir}/id_mappings.dgl", id_mappings)

def fill_part_features(input_dir, output_dir, part_id, num_features):
    part_data = dgl.data.load_tensors(f"{input_dir}/part{part_id}/data.dgl")
    num_nodes_in_part = part_data['inner_node'].sum().tolist()
    dgl.data.save_tensors(f"{output_dir}/part{part_id}/node_feat.dgl", {
        "_N/features": torch.rand(num_nodes_in_part, num_features)
    })

    dgl.data.save_tensors(f"{output_dir}/part{part_id}/edge_feat.dgl", {})

def fill_features(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_parts = args.num_parts
    graph_name = args.graph_name
    num_features = args.num_features

    torch.manual_seed(args.random_seed)
    fill_infer_targets_features(output_dir, num_features)
    for i in range(num_parts):
        fill_part_features(input_dir, output_dir, i, num_features)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Partition builtin graphs")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_parts', type=int, required=True)
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--num_features', type=int)
    parser.add_argument('--random_seed', type=int, default=5125323)
    parser.add_argument('--stage', type=str, choices=["save_graphs", "fill_features"])

    args = parser.parse_args()
    print(args)

    if args.stage == "save_graphs":
        save_graphs(args)
    elif args.stage == "fill_features":
        fill_features(args)
