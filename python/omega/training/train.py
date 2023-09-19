import argparse
from pathlib import Path
import json

import torch
import numpy as np
from sklearn.metrics import f1_score

import dgl
from omega.utils import load_graph
from omega.utils import get_dataset_config
from omega.models import create_model

def do_full_training(args, g, model, device, dataset_config, result_dir):
    model_path = result_dir / "model.pt"

    best_val_f1 = -1
    num_layers = args.num_layers
    multilabel = dataset_config.multilabel

    if multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_mask = g.ndata["train_mask"].type(torch.bool)
    val_mask = g.ndata["val_mask"].type(torch.bool)
    test_mask = g.ndata["test_mask"].type(torch.bool)

    if dataset_config.inductive:
        nids = torch.arange(g.number_of_nodes()).to(device)
        train_nids = torch.masked_select(nids, train_mask)
        train_g = g.subgraph(train_nids)
    else:
        train_g = g

    blocks = [train_g] * args.num_layers

    for epoch in range(args.num_epochs):
        model.train()
        logits = model(blocks, train_g.ndata["features"])
        labels = train_g.ndata["labels"]

        if not dataset_config.inductive:
            logits = logits[train_mask]
            labels = labels[train_mask]

        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.val_every == 0:
            val_f1_mic, val_f1_mac = evaluate(model, g, val_mask, num_layers, multilabel)
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Val F1_mic {val_f1_mic:.4f} | Val F1_mac {val_f1_mac:.4f}")
            if val_f1_mic > best_val_f1:
                best_val_f1 = val_f1_mic
                print("Found best validation f1_mic.")
                # TODO(gwkim): Save the best model.
                torch.save(model.state_dict(), model_path)
        else:
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} |")

    model.load_state_dict(torch.load(model_path))
    f1_mic, f1_mac = evaluate(model, g, test_mask, num_layers, multilabel)
    print(f"Test F1_mic {f1_mic:.4f} | Test F1_mac {f1_mac:.4f}")
    return f1_mic

def do_sampled_training(args, g, model, device, fanouts, dataset_config, result_dir):
    model_path = result_dir / "model.pt"

    best_val_f1 = -1

    num_layers = args.num_layers
    multilabel = dataset_config.multilabel

    if dataset_config.multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_mask = g.ndata["train_mask"].type(torch.bool)
    val_mask = g.ndata["val_mask"].type(torch.bool)
    test_mask = g.ndata["test_mask"].type(torch.bool)

    if dataset_config.inductive:
        nids = torch.arange(g.number_of_nodes()).to(device)
        train_nids = torch.masked_select(nids, train_mask)
        train_g = g.subgraph(train_nids)
    else:
        train_g = g

    train_loader = dgl.dataloading.DataLoader(
        train_g,
        train_g.ndata["train_mask"].nonzero().reshape(-1),
        dgl.dataloading.MultiLayerNeighborSampler(fanouts),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    for epoch in range(args.num_epochs):
        model.train()

        for input_nids, seeds, blocks in train_loader:
            input_features = train_g.ndata["features"][input_nids]
            logits = model(blocks, input_features)
            labels = train_g.ndata["labels"][seeds]
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.val_every == 0:
            val_f1_mic, val_f1_mac = evaluate(model, g, val_mask, num_layers, multilabel)
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Val F1_mic {val_f1_mic:.4f} | Val F1_mac {val_f1_mac:.4f}")
            if val_f1_mic > best_val_f1:
                best_val_f1 = val_f1_mic
                print("Found best validation f1_mic.")
                # TODO(gwkim): Save the best model.
                torch.save(model.state_dict(), model_path)
        else:
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} |")

    model.load_state_dict(torch.load(model_path))
    f1_mic, f1_mac = evaluate(model, g, test_mask, num_layers, multilabel)
    print(f"Test F1_mic {f1_mic:.4f} | Test F1_mac {f1_mac:.4f}")
    return f1_mic

def evaluate(model, g, mask, num_layers, multilabel):

    with torch.no_grad():
        model.eval()
        logits = model([g] * num_layers, g.ndata["features"])
        labels = g.ndata["labels"]

        logits = logits[mask].to("cpu")
        labels = labels[mask].to("cpu")
        return cal_metrics(labels.cpu().numpy(),
                    logits.cpu().numpy(), multilabel)

def cal_metrics(y_true, y_pred, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
        return f1_score(y_true, y_pred, average="micro"), \
            f1_score(y_true, y_pred, average="macro")
    else:
        y_pred = np.argmax(y_pred, axis=1)
        return f1_score(y_true, y_pred, average="micro"), \
            f1_score(y_true, y_pred, average="macro")

def main(args):
    device = f"cuda:{args.local_rank}"
    dataset_config = get_dataset_config(args.graph_name)
    g = load_graph(
        args.graph_name,
        ogbn_data_root=args.ogbn_data_root,
        saint_data_root=args.saint_data_root
    ).to(device)

    gat_heads = [int(h) for h in args.gat_heads.split(",")] if args.gat_heads else []
    if args.gnn == "gat":
        assert len(gat_heads) == args.num_layers
        assert all([h > 0 for h in gat_heads])

    model = create_model(
        args.gnn,
        dataset_config.num_inputs,
        args.num_hiddens,
        dataset_config.num_classes,
        args.num_layers,
        gat_heads,
        dropout=args.dropout).to(device)

    fanouts = [int(f) for f in args.fanouts.split(",")] if args.fanouts else [-1] * args.num_layers
    fanouts = [-1 if f == 0 else f for f in fanouts]
    if fanouts[0] == -1:
        full_training = True
        assert all([f == -1 for f in fanouts])
    else:
        full_training = False
        assert all([f > 0 for f in fanouts])

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if full_training:
        test_f1_mic = do_full_training(args, g, model, device, dataset_config, result_dir)
    else:
        test_f1_mic = do_sampled_training(args, g, model, device, fanouts, dataset_config, result_dir)
    
    args_dict = vars(args)
    args_dict["test_f1_mic"] = test_f1_mic

    with open(result_dir / "config.json", "w") as f:
        f.write(json.dumps(args_dict, indent=4, sort_keys=True))
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=10) # For fixed_epochs
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--val_every", default=10, type=int)

    parser.add_argument('--graph_name', type=str, default='reddit',
                        help='datasets: reddit, ogbn-products, ogbn-papers100M, amazon, yelp, flickr')
    parser.add_argument('--ogbn_data_root', type=str)
    parser.add_argument('--saint_data_root', type=str)

    parser.add_argument('--gnn', type=str, required=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_hiddens', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--gat_heads', type=str)
    parser.add_argument('--fanouts', type=str)

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=5123412)
    args = parser.parse_args()
    main(args)
