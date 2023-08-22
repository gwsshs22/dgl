import argparse

import torch
import numpy as np

import dgl

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    dgl.distributed.initialize(
        args.ip_config,
        net_type=args.net_type,
        load_precoms=args.use_precoms,
        num_layers=args.num_layers,
        num_hiddens=args.num_hiddens,
        precom_path=args.precom_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--net_type', type=str, default='socket',
                        help="backend net type, 'socket' or 'tensorpipe'")
    parser.add_argument("--use_precoms", action="store_true")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_hiddens", type=int)
    parser.add_argument("--precom_path", type=str, default="")

    parser.add_argument('--random_seed', type=int, default=5123412)
    args = parser.parse_args()

    main(args)
