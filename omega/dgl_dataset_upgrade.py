import argparse
import json
from pathlib import Path

import dgl


def main(args):
    path = Path(args.path)
    conf_file_path = (path / f"{args.graph_name}.json")
    
    with conf_file_path.open(mode='r') as f:
        # load the JSON data from the file
        part_metadata = json.load(f)
        num_parts = part_metadata["num_parts"]
    if "_E" in part_metadata["etypes"]:
        del part_metadata["etypes"]["_E"]
        part_metadata["etypes"]["_N:_E:_N"] = 0

    with conf_file_path.open('w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

    num_parts = part_metadata["num_parts"]
    
    for i in range(num_parts):
        edge_feat_path = path / f"part{i}" / "edge_feat.dgl"
        data = dgl.data.load_tensors(str(edge_feat_path))
        new_data = {}
        for k, v in data.items():
            key = k if k != "_E" else "_N:_E:_N"
            new_data[k] = v
        dgl.data.save_tensors(str(edge_feat_path), new_data)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--graph_name', type=str, required=True)

    args = parser.parse_args()
    main(args)
