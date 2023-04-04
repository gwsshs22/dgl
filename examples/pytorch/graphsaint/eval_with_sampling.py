import argparse
import os
import time
import warnings
import dgl
import torch
import torch.nn.functional as F
from config import CONFIG
from modules import GCNNet
from sampler import SAINTEdgeSampler, SAINTNodeSampler, SAINTRandomWalkSampler
from torch.utils.data import Dataset, DataLoader
from utils import Logger, calc_f1, evaluate, load_data, save_log_dir
import numpy as np
from tqdm import tqdm


# def sampling_eval_graph(g, device, ratio):
#     test_nids = g.ndata["test_mask"].nonzero().squeeze()
#     subg1 = g.sample_neighbors(test_nids, -1)
#     block1 = dgl.to_block(subg1, test_nids)

#     subg2 = g.sample_neighbors(block1.srcdata[dgl.NID], -1)
#     block2 = dgl.to_block(subg2, block1.srcdata[dgl.NID])

#     testg_eids = torch.concat((subg1.edata[dgl.NID], subg2.edata[dgl.NID])).unique()
#     testg = g.edge_subgraph(testg_eids)

#     test_local_ids = testg.ndata["test_mask"].nonzero().squeeze().cpu()
#     existing_local_ids = torch.logical_not(testg.ndata["test_mask"]).nonzero().squeeze().cpu()

#     num_included = int(existing_local_ids.shape[0] * ratio)
#     included_existing_local_ids = dgl.random.choice(existing_local_ids, num_included, replace=False)
#     print(included_existing_local_ids.shape)


#     all_local_ids = torch.concat((test_local_ids, included_existing_local_ids))
#     t = all_local_ids.shape[0]
#     all_local_ids = all_local_ids.unique()
#     assert(t == all_local_ids.shape[0])

#     return testg.subgraph(all_local_ids.to(device))

# def sampling_eval_graph(g, device, ratio):
#     test_nids = g.ndata["test_mask"].nonzero().squeeze()
#     subg1 = g.sample_neighbors(test_nids, -1)
#     block1 = dgl.to_block(subg1, test_nids)

#     subg2 = g.sample_neighbors(block1.srcdata[dgl.NID], -1)
#     block2 = dgl.to_block(subg2, block1.srcdata[dgl.NID])

#     testg_eids = torch.concat((subg1.edata[dgl.NID], subg2.edata[dgl.NID])).unique()
#     testg = g.edge_subgraph(testg_eids)

#     test_local_ids = testg.ndata["test_mask"].nonzero().squeeze().cpu()
#     test_local_ids = dgl.random.choice(test_local_ids, 128, replace=False)

#     other_local_ids = torch.tensor(np.setdiff1d(np.arange(testg.number_of_nodes()), test_local_ids.numpy()))
#     num_included = int(other_local_ids.shape[0] * ratio)

#     included_other_local_ids = dgl.random.choice(
#         other_local_ids,
#         num_included,
#         replace=False)
#     print(included_other_local_ids.shape)


#     all_local_ids = torch.concat((test_local_ids, included_other_local_ids))
#     t = all_local_ids.shape[0]
#     all_local_ids = all_local_ids.unique()
#     assert(t == all_local_ids.shape[0])

#     return testg.subgraph(all_local_ids.to(device))

# def sampling_eval_graph(g, device, ratio):
#     # test_nids = g.ndata["test_mask"].nonzero().squeeze()
#     test_nids = g.ndata["test_mask"].nonzero().squeeze()
#     test_nids = dgl.random.choice(test_nids.cpu(), 128, replace=False).to(device)
#     subg1 = g.sample_neighbors(test_nids, -1)
#     block1 = dgl.to_block(subg1, test_nids)

#     subg2 = g.sample_neighbors(block1.srcdata[dgl.NID], -1)
#     block2 = dgl.to_block(subg2, block1.srcdata[dgl.NID])

#     testg_eids = torch.concat((subg1.edata[dgl.NID], subg2.edata[dgl.NID])).unique()
#     testg = g.edge_subgraph(testg_eids)

#     test_local_ids = testg.ndata["test_mask"].nonzero().squeeze().cpu()
#     # test_local_ids = dgl.random.choice(test_local_ids, 128, replace=False)

#     other_local_ids = np.setdiff1d(np.arange(testg.number_of_nodes()), test_local_ids.numpy())
#     in_degrees = testg.in_degrees().type(torch.float64).clamp(min=1)[other_local_ids].cpu().numpy()
#     in_degrees = (in_degrees / in_degrees.sum())

#     num_included = int(other_local_ids.shape[0] * ratio)

#     included_other_local_ids = np.random.choice(other_local_ids, num_included, replace=False, p=in_degrees)
#     included_other_local_ids = torch.tensor(included_other_local_ids)

#     all_local_ids = torch.concat((test_local_ids, included_other_local_ids))
#     t = all_local_ids.shape[0]
#     all_local_ids = all_local_ids.unique()
#     assert(t == all_local_ids.shape[0])

#     return testg.subgraph(all_local_ids.to(device))

class IndexTensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def sampling_eval_graph(g, target_test_nids, device, ratio):
    target_mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool).to(device)
    target_mask[target_test_nids] = True
    g.ndata["target_mask"] = target_mask

    # Only include 2-hop neighbors of the target test nodes
    subg1 = g.sample_neighbors(target_test_nids, -1)
    block1 = dgl.to_block(subg1, target_test_nids)
    subg2 = g.sample_neighbors(block1.srcdata[dgl.NID], -1)
    block2 = dgl.to_block(subg2, block1.srcdata[dgl.NID])

    
    testg_eids = torch.concat((subg1.edata[dgl.NID], subg2.edata[dgl.NID])).unique()
    testg = g.edge_subgraph(testg_eids)
    
    target_test_nids = testg.ndata["target_mask"].nonzero().squeeze() 
    existing_nids = torch.logical_not(testg.ndata["target_mask"]).nonzero().squeeze()

    in_degrees = testg.in_degrees()[existing_nids].type(torch.float64).clamp(min=1)
    in_degrees = (in_degrees / in_degrees.sum()).cpu()

    num_included = int(existing_nids.shape[0] * ratio)
    included_existing_nids = np.random.choice(existing_nids.cpu(), num_included, replace=False, p=in_degrees)
    included_existing_nids = torch.tensor(included_existing_nids).to(device)

    return testg.subgraph(torch.concat((target_test_nids, included_existing_nids)))

def convert_to_block(g, seeds, features, labels, device, fanouts):
    blocks = []
    for fanout in fanouts:
        frontier = local_sample_neighbors(g, seeds.to(device), fanout)
        block = dgl.to_block(frontier).to(device)
        seeds = block.srcdata[dgl.NID]
        blocks.insert(0, block)
    return blocks, features[seeds].to(device), labels[seeds].to(device)

def main(args, task, ratio):
    warnings.filterwarnings("ignore")
    multilabel_data = {"ppi", "yelp", "amazon"}
    multilabel = args.dataset in multilabel_data

    # This flag is excluded for too large dataset, like amazon, the graph of which is too large to be directly
    # shifted to one gpu. So we need to
    # 1. put the whole graph on cpu, and put the subgraphs on gpu in training phase
    # 2. put the model on gpu in training phase, and put the model on cpu in validation/testing phase
    # We need to judge cpu_flag and cuda (below) simultaneously when shift model between cpu and gpu
    if args.dataset in ["amazon"]:
        cpu_flag = True
    else:
        cpu_flag = False

    # load and preprocess dataset
    data = load_data(args, multilabel)
    g = data.g
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    train_nid = data.train_nid

    in_feats = g.ndata["feat"].shape[1]
    n_classes = data.num_classes
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print(
        """----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes/Labels (multi binary labels) %d
    #Train samples %d
    #Val samples %d
    #Test samples %d"""
        % (
            n_nodes,
            n_edges,
            n_classes,
            n_train_samples,
            n_val_samples,
            n_test_samples,
        )
    )
    # load sampler

    kwargs = {
        "dn": args.dataset,
        "g": g,
        "train_nid": train_nid,
        "num_workers_sampler": args.num_workers_sampler,
        "num_subg_sampler": args.num_subg_sampler,
        "batch_size_sampler": args.batch_size_sampler,
        "online": args.online,
        "num_subg": args.num_subg,
        "full": args.full,
    }

    device = "cuda:{}".format(args.gpu)
    g = g.to(device)


    labels = g.ndata["label"]
    print("labels shape:", g.ndata["label"].shape)
    print("features shape:", g.ndata["feat"].shape)

    model = GCNNet(
        in_dim=in_feats,
        hid_dim=args.n_hidden,
        out_dim=n_classes,
        arch=args.arch,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
        aggr=args.aggr)
    model.cuda()
    model.eval()

    # test
    log_dir = save_log_dir(args)
    model.load_state_dict(
        torch.load(os.path.join(log_dir, "best_model_{}.pkl".format(task)))
    )

    if cpu_flag and cuda:
        model = model.to("cpu")

    # with torch.no_grad():
    #     g.ndata["full_D_norm"] = 1.0 / g.in_degrees().float().clamp(
    #         min=1
    #     ).unsqueeze(1)
    #     f1_mic, f1_mac = evaluate(model, g, labels, test_mask, multilabel)
    #     print(
    #         "Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(
    #             f1_mic, f1_mac
    #         )
    #     )
        # logits = model(g)
        # logits = logits[g.ndata["test_mask"]]
        # labels = g.ndata["label"][g.ndata["test_mask"]]
        # test_f1_mic, test_f1_mac = calc_f1(
        #     labels.cpu().numpy(), logits.cpu().numpy(), multilabel
        # )
        # print(
        #     "Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(
        #         test_f1_mic, test_f1_mac
        #     )
        # )
    g.ndata["full_D_norm"] = 1.0 / g.in_degrees().float().clamp(
        min=1
    ).unsqueeze(1)

    # num_test_nodes = 256
    # test_nids = g.ndata["test_mask"].nonzero().squeeze().cpu()
    # target_test_nids = dgl.random.choice(test_nids, num_test_nodes, replace=False).to(device)
    # test_g = sampling_eval_graph(g, target_test_nids, device, ratio)
    data_loader = DataLoader(IndexTensorDataset(g.ndata["test_mask"].nonzero().squeeze()),
        batch_size=256, shuffle=True)

    logits_arr = []
    labels_arr = []

    i = 0
    with torch.no_grad():
        for target_test_nids in tqdm(data_loader):
            test_g = sampling_eval_graph(g, target_test_nids, device, ratio)
            test_g.ndata["full_D_norm"] = 1.0 / test_g.in_degrees().float().clamp(
                min=1
            ).unsqueeze(1)

            logits = model(test_g)
            logits = logits[test_g.ndata["target_mask"]]
            labels = test_g.ndata["label"][test_g.ndata["target_mask"]]

            logits_arr.append(logits)
            labels_arr.append(labels)

            test_f1_mic, test_f1_mac = calc_f1(
                labels.cpu().numpy(), logits.cpu().numpy(), multilabel
            )

    logits = torch.cat(logits_arr).cpu().numpy()
    labels = torch.cat(labels_arr).cpu().numpy()

    test_f1_mic, test_f1_mac = calc_f1(labels, logits, multilabel)

    print(
        "Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(
            test_f1_mic, test_f1_mac
        )
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="GraphSAINT")
    parser.add_argument(
        "--task", type=str, default="ppi_n", help="type of tasks"
    )
    parser.add_argument(
        "--online",
        dest="online",
        action="store_true",
        help="sampling method in training phase",
    )
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument("--ratio", type=float, default=0.0, help="The ratio of sampled computation graph for inference nodes")
    task = parser.parse_args().task
    ratio = parser.parse_args().ratio
    args = argparse.Namespace(**CONFIG[task])
    args.online = parser.parse_args().online
    args.gpu = parser.parse_args().gpu
    print(args)

    main(args, task=task, ratio=ratio)
