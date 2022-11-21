import argparse
import random
import json
import os

import torch
import numpy as np
import dgl
from dgl import backend as F

def get_split(num_parts, size):
    ret = [size // num_parts] * num_parts
    for i in range(size % num_parts):
        ret[i] += 1
    return ret

def main(args):
    graph_name = args.graph_name
    part_config = args.part_config
    config_path = os.path.dirname(part_config)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    print(part_metadata)
    assert part_metadata["part_method"] == "random", "P^3 should be used with random partitioning"

    num_parts = part_metadata["num_parts"]
    
    for part_id in range(num_parts):
        node_feats = dgl.data.utils.load_tensors(os.path.join(config_path, f"part{part_id}", "node_feat.dgl"))
        node_feats = node_feats["_N/features"]
        split = get_split(num_parts, node_feats.shape[1])
        splitted_node_feats = node_feats.split(split, dim=1)

        for split_idx in range(len(splitted_node_feats)):
            tensor_file_path = os.path.join(config_path, f"feat_split_{split_idx}_part_{part_id}.dgl")
            tensor_dict = {"feat": splitted_node_feats[split_idx]}
            dgl.data.utils.save_tensors(tensor_file_path, tensor_dict)

    for split_idx in range(num_parts):
        tensors = []
        for part_id in range(num_parts):
            tensor_file_path = os.path.join(config_path, f"feat_split_{split_idx}_part_{part_id}.dgl")
            tensor = F.zerocopy_from_dgl_ndarray(dgl.data.utils.load_tensors(tensor_file_path, tensor_dict)["feat"])
            tensors.append(tensor)
            os.remove(tensor_file_path)
        tensor = torch.concat(tensors)
        dgl.data.utils.save_tensors(os.path.join(config_path, f"part{split_idx}", f"p3_features.dgl"), { "features": tensor })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', type=str, required=True)
    parser.add_argument('--part_config', type=str, required=True)

    args = parser.parse_args()
    print(args)

    main(args)
