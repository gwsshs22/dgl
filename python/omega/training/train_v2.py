import argparse
from pathlib import Path
import json
import secrets
import sys
import random

import gc
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

import dgl
from dgl import function as fn
from omega.utils import load_graph, get_dataset_config, cal_labels, cal_metrics, get_graph_data_root, get_pna_delta
from omega.models import create_model

def saint_preprocess(args, train_g, device, num_roots, walk_length, max_iter=50000):
    saint_data_path = get_graph_data_root(
        args.graph_name,
        ogbn_data_root=args.ogbn_data_root,
        saint_data_root=args.saint_data_root
    ) / f"saint_{num_roots}_{walk_length}.th"

    if saint_data_path.exists():
        print(f"Found a GraphSAINT state on {saint_data_path}.", file=sys.stderr)
        saint_state = torch.load(saint_data_path)
        return saint_state["N"], saint_state["l_n"], saint_state["a_n"], saint_state["subg_node_ids"]

    subg_node_ids = []

    saint_g = train_g.to("cpu")
    for nk in [k for k, v in saint_g.ndata.items()]:
        saint_g.ndata.pop(nk)
    for ek in [k for k, v in saint_g.edata.items()]:
        saint_g.edata.pop(ek)
    num_total_nodes = saint_g.num_nodes()
    num_total_edges = saint_g.num_edges()

    sampler = dgl.dataloading.SAINTSampler(mode='walk', budget=(num_roots, walk_length))

    node_counter = torch.zeros((num_total_nodes,))
    edge_counter = torch.zeros((num_total_edges,))
    N = 0
    num_subg_nodes = 0
    print("Start preprocessing GraphSAINT state.", file=sys.stderr)
    for _ in range(max_iter):
        subg = sampler.sample(saint_g, None)
        subg_node_ids.append(subg.ndata[dgl.NID])

        node_counter[subg.ndata[dgl.NID]] += 1
        edge_counter[subg.edata[dgl.NID]] += 1
        N += 1
        num_subg_nodes += subg.num_nodes()

        if N >= 50 * num_total_nodes / (num_subg_nodes / N):
            break

    print(f"Done preprocessing GraphSAINT state. N={N}", file=sys.stderr)
    if N == max_iter:
        print(f"[WARN] GraphSAINT preprocessing sampling iterations reaches max_iter of {max_iter}", file=sys.stderr)

    node_counter[node_counter == 0] = 1
    edge_counter[edge_counter == 0] = 1

    loss_norm = N / node_counter / num_total_nodes

    saint_g.ndata["n_c"] = node_counter
    saint_g.edata["e_c"] = edge_counter
    saint_g.apply_edges(fn.v_div_e("n_c", "e_c", "a_n"))
    aggr_norm = saint_g.edata.pop("a_n")

    torch.save({
        "N": N,
        "l_n": loss_norm,
        "a_n": aggr_norm,
        "subg_node_ids": subg_node_ids
    }, saint_data_path)

    return N, loss_norm, aggr_norm, subg_node_ids

def do_training(args, g, model, device, dataset_config, result_dir, val_every, fanouts):
    use_gcn_both_norm = (args.gnn == "gcn" or args.gnn == "gcn2") and args.gcn_norm == "both"
    model_path = result_dir / "model.pt"

    best_val_f1 = -1

    large_graph = dataset_config.large
    inductive = args.force_inductive or dataset_config.inductive
    num_layers = args.num_layers
    multilabel = dataset_config.multilabel

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if large_graph:
        if inductive:
            train_nids = g.ndata["train_mask"].type(torch.bool).nonzero().reshape(-1)
            train_g = g.subgraph(train_nids)
            train_g = train_g.to(device)
        else:
            train_g = g
    else:
        g = g.to(device)
        
        if inductive:
            train_nids = g.ndata["train_mask"].type(torch.bool).nonzero().reshape(-1)
            train_g = g.subgraph(train_nids)
        else:
            train_g = g

    if args.sampling_method == "full":
        if dataset_config.multilabel:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        train_mask = train_g.ndata["train_mask"].to(device)
        def run_epoch():
            logits = model([train_g] * args.num_layers, train_g.ndata["features"])
            labels = train_g.ndata["labels"]

            if not inductive:
                logits = logits[train_mask]
                labels = labels[train_mask]

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

    elif args.sampling_method == "ns":
        # Train with neighborhood sampling

        # train_mask = train_g.ndata["train_mask"]
        # train_g_device = train_mask.device
        # train_loader = torch.utils.data.DataLoader(
        #     torch.masked_select(torch.arange(train_mask.shape[0]), train_mask),
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     drop_last=False)

        train_loader = dgl.dataloading.DataLoader(
            train_g,
            train_g.ndata["train_mask"].nonzero().reshape(-1),
            dgl.dataloading.MultiLayerNeighborSampler(fanouts),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False
        )

        if dataset_config.multilabel:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        # def run_epoch():

        #     for batch_id, seeds in enumerate(train_loader):
        #         s = seeds
        #         blocks = []

        #         if large_graph:
        #             gc.collect()
        #             torch.cuda.empty_cache()

        #         for fanout in fanouts:
        #             f = dgl.sampling.sample_neighbors(train_g, s, fanout)
        #             f = f.to(device)
        #             block = dgl.to_block(f, s)
        #             s = block.srcdata[dgl.NID]
        #             s = s.to(train_g_device)

        #             if use_gcn_both_norm:
        #                 block.set_out_degrees(train_g.out_degrees(s).to(device))
                    
        #             blocks.insert(0, block)

        #         input_features = train_g.ndata["features"][blocks[0].srcdata[dgl.NID].to(train_g_device)]

        #         print(f"({batch_id}/{len(train_loader)}){blocks}", file=sys.stderr)
        #         input_features = input_features.to(device)

        #         h = model.feature_preprocess(input_features)
        #         h0 = h
        #         for layer_idx, block in enumerate(blocks):
        #             h = model.layer_foward(layer_idx, block.to(device), h, h0)
        #         logits = h

        #         labels = train_g.ndata["labels"][seeds].to(device)

        #         loss = loss_fn(logits, labels)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #     return loss.item()

        def run_epoch():
            for input_nids, seeds, blocks in train_loader:
                input_features = train_g.ndata["features"][input_nids]

                if use_gcn_both_norm:
                    for b in blocks:
                        b.set_out_degrees(train_g.out_degrees(b.srcdata[dgl.NID]))

                input_features = input_features.to(device)

                h = model.feature_preprocess(input_features)
                h0 = h
                for layer_idx, block in enumerate(blocks):
                    h = model.layer_foward(layer_idx, block.to(device), h, h0)
                logits = h

                labels = train_g.ndata["labels"][seeds].to(device)

                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return loss.item()
    elif args.sampling_method == "saint" and args.use_saint_norm:
        # Train with GraphSAINT-based subgraph sampling with normalization
        assert args.num_roots is not None
        num_roots = args.num_roots
        walk_length = args.walk_length if args.walk_length is not None else num_layers
        N, loss_norm, aggr_norm, subg_node_ids = saint_preprocess(args, train_g, device, num_roots, walk_length)
        train_g.ndata["l_n"] = loss_norm.to(device)
        train_g.edata["a_n"] = aggr_norm.to(device)


        subg_idx = 0
        holder = [subg_idx]
        def run_epoch():
            root_nids = subg_node_ids[holder[0]].to(device)
            holder[0] += 1
            if holder[0] == len(subg_node_ids):
                random.shuffle(subg_node_ids)
                holder[0] = 0

            subg = train_g.subgraph(root_nids.to(device), relabel_nodes=True)

            logits = model([subg] * num_layers, subg.ndata["features"], saint_normalize=True)
            labels = subg.ndata["labels"]

            if dataset_config.multilabel:
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="sum",
                    weight=subg.ndata["l_n"].unsqueeze(1),
                )
            else:
                loss = F.cross_entropy(logits, labels, reduction="none")
                loss = (subg.ndata["l_n"] * loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss.item()
    elif args.sampling_method == "saint" and not args.use_saint_norm:
        # Train with GraphSAINT-based subgraph sampling without normalization
        walk_length = args.walk_length if args.walk_length is not None else num_layers

        train_g = train_g.to("cpu")
        train_mask = train_g.ndata["train_mask"]
        train_loader = torch.utils.data.DataLoader(
            torch.masked_select(torch.arange(train_mask.shape[0]), train_mask),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False)

        if dataset_config.multilabel:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        def run_epoch():
            for root_nids in train_loader:
                traces, types = dgl.sampling.random_walk(train_g, nodes=root_nids, length=walk_length)
                sampled_nodes, _, _, _ = dgl.sampling.pack_traces(traces, types)

                subg = train_g.subgraph(sampled_nodes).to(device)

                logits = model([subg] * num_layers, subg.ndata["features"])
                labels = subg.ndata["labels"]

                logits = logits[subg.ndata["train_mask"]]
                labels = labels[subg.ndata["train_mask"]]
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            return loss

    if large_graph:
        val_nids = g.ndata["val_mask"].type(torch.bool).nonzero().reshape(-1)

        val_blocks = []
        seeds = val_nids
        for _ in range(num_layers):
            f = dgl.sampling.sample_neighbors(g, seeds, -1)
            val_block = dgl.to_block(f, seeds)

            val_blocks.insert(0, val_block)
            seeds = val_block.srcdata[dgl.NID]
            if use_gcn_both_norm:
                val_block.set_out_degrees(g.out_degrees(seeds))
    else:
        val_blocks = None

    patience = args.patience
    for epoch in range(args.num_epochs):
        if patience < 0:
            break

        model.train()
        loss = run_epoch()
        gc.collect()
        torch.cuda.empty_cache()

        if (epoch + 1) % val_every == 0:
            val_f1_mic, val_f1_mac, _ = evaluate(model, g, device, g.ndata["val_mask"], num_layers, multilabel, large_graph, use_gcn_both_norm, blocks=val_blocks)
            print(f"Epoch {epoch:05d} | Loss {loss:.4f} | Val F1_mic {val_f1_mic:.4f} | Val F1_mac {val_f1_mac:.4f}")
            if val_f1_mic > best_val_f1:
                best_val_f1 = val_f1_mic
                patience = args.patience
                print("Found best validation f1_mic.")
                # TODO(gwkim): Save the best model.
                torch.save(model.state_dict(), model_path)
            else:
                patience -= 1
        else:
            print(f"Epoch {epoch:05d} | Loss {loss:.4f} |")
        gc.collect()
        torch.cuda.empty_cache()

    del train_g
    gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(model_path))
    f1_mic, f1_mac, test_logits = evaluate(model, g, device, g.ndata["test_mask"], num_layers, multilabel, large_graph, use_gcn_both_norm)
    print(f"Test F1_mic {f1_mic:.4f} | Test F1_mac {f1_mac:.4f}")
    return f1_mic, epoch, test_logits

def evaluate(model, g, device, mask, num_layers, multilabel, large_graph, use_gcn_both_norm, blocks=None):
    with torch.no_grad():
        mask = mask.type(torch.bool)
        model.eval()
        if large_graph:
            if blocks is None:
                evaluate_nids = mask.type(torch.bool).nonzero().reshape(-1)

                blocks = []
                seeds = evaluate_nids
                for _ in range(num_layers):
                    f = dgl.sampling.sample_neighbors(g, seeds, -1)
                    block = dgl.to_block(f, seeds)

                    blocks.insert(0, block)
                    seeds = block.srcdata[dgl.NID]
                    if use_gcn_both_norm:
                        block.set_out_degrees(g.out_degrees(seeds))

            assert(len(blocks) == num_layers)

            h = model.feature_preprocess(
                g.ndata["features"][blocks[0].srcdata[dgl.NID]].to(device))
            h0 = h
            for layer_idx, block in enumerate(blocks):
                h = model.layer_foward(layer_idx, block.to(device), h, h0)
            logits = h.to("cpu")
            labels = g.ndata["labels"][mask].to("cpu")

            predicted_labels = cal_labels(logits.cpu().numpy(), multilabel)
            return f1_score(labels.numpy(), predicted_labels, average="micro"), f1_score(labels.numpy(), predicted_labels, average="macro"), torch.tensor(predicted_labels)
        else:
            logits = model([g] * num_layers, g.ndata["features"])
            labels = g.ndata["labels"]

            logits = logits[mask].to("cpu")
            labels = labels[mask].to("cpu")

            predicted_labels = cal_labels(logits.cpu().numpy(), multilabel)
            return f1_score(labels.numpy(), predicted_labels, average="micro"), f1_score(labels.numpy(), predicted_labels, average="macro"), torch.tensor(predicted_labels)

def main(args):
    device = f"cuda:{args.local_rank}"
    dataset_config = get_dataset_config(args.graph_name)
    g = load_graph(
        args.graph_name,
        ogbn_data_root=args.ogbn_data_root,
        saint_data_root=args.saint_data_root
    )

    gat_heads = [int(h) for h in args.gat_heads.split(",")] if args.gat_heads else []
    if args.gnn == "gat":
        assert len(gat_heads) == args.num_layers
        assert all([h > 0 for h in gat_heads])
    
    if args.gnn == 'pna':
        pna_delta = get_pna_delta(g)
    else:
        pna_delta = torch.tensor(-1.0)

    model = create_model(
        args.gnn,
        dataset_config.num_inputs,
        args.num_hiddens,
        dataset_config.num_classes,
        args.num_layers,
        gat_heads,
        gcn_norm=args.gcn_norm,
        dropout=args.dropout,
        gcn2_alpha=args.gcn2_alpha,
        sage_aggr=args.sage_aggr,
        pna_delta=pna_delta).to(device)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    val_every = args.val_every if args.val_every else 1

    fanouts = None
    if args.sampling_method == "full":
        val_every = args.val_every if args.val_every else 10
    elif args.sampling_method == "ns":
        assert args.fanouts is not None
        fanouts = [int(f) for f in args.fanouts.split(",")] if args.fanouts else [-1] * args.num_layers
        fanouts = [f if f != 0 else -1 for f in fanouts]
        if fanouts[0] == -1:
            assert(all([f == -1 for f in fanouts]))
        else:
            assert(all([f > 0 for f in fanouts]))

        assert len(fanouts) == args.num_layers
    elif args.sampling_method == "saint":
        if args.use_saint_norm:
            assert args.num_roots is not None

    test_f1_mic, running_epochs, predicted_test_labels = do_training(args, g, model, device, dataset_config, result_dir, val_every, fanouts)

    args_dict = vars(args)
    args_dict["test_f1_mic"] = test_f1_mic
    args_dict["running_epochs"] = running_epochs
    args_dict["val_every"] = val_every
    args_dict["fanouts"] = ",".join([str(f) for f in fanouts]) if fanouts is not None else ""
    args_dict["num_roots"] = args.num_roots if args.num_roots is not None else -1
    args_dict["id"] = secrets.token_hex(10)
    args_dict["pna_delta"] = pna_delta.item()

    with open(result_dir / "config.json", "w") as f:
        f.write(json.dumps(args_dict, indent=4, sort_keys=True))
        f.write("\n")

    torch.save(predicted_test_labels, result_dir / "predicted_test_labels.th")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=10) # For fixed_epochs
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--val_every", type=int)

    parser.add_argument('--graph_name', type=str, default='reddit',
                        help='datasets: reddit, ogbn-products, ogbn-papers100M, amazon, yelp, flickr')
    parser.add_argument('--ogbn_data_root', type=str)
    parser.add_argument('--saint_data_root', type=str)

    parser.add_argument('--gnn', type=str, required=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gcn2_alpha', type=float, default=0.5)
    parser.add_argument('--num_hiddens', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--gat_heads', type=str)
    parser.add_argument('--gcn_norm', type=str, default='both', choices=['right', 'both'])
    parser.add_argument('--sage_aggr', type=str, default='mean', choices=['mean', 'lstm'])
    parser.add_argument('--sampling_method', type=str, choices=['full', 'ns', 'saint'])
    parser.add_argument('--force_inductive', action='store_true')

    parser.add_argument('--fanouts', type=str)

    parser.add_argument('--num_roots', type=int)
    parser.add_argument('--walk_length', type=int)
    parser.add_argument('--use_saint_norm', action='store_true')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=5123412)
    args = parser.parse_args()
    print(f"Training args={args}")
    main(args)
