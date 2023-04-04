import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import f1_score
from dgl.sampling import sample_neighbors as local_sample_neighbors
import dgl

class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[1], out_size, heads[2], residual=True, activation=None))
        
    def forward(self, blocks, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(blocks[i], h)
            if i == 2:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
        return h

def convert_to_block(g, seeds, features, labels, device, fanouts):
    blocks = []
    for fanout in fanouts:
        frontier = local_sample_neighbors(g, seeds.to(device), fanout)
        block = dgl.to_block(frontier).to(device)
        seeds = block.srcdata[dgl.NID]
        blocks.insert(0, block)

    return blocks, features[seeds].to(device), labels[seeds].to(device)

def evaluate(g, features, labels, model):
    model.eval()
    with torch.no_grad():
        output = model([g, g, g], features)
        pred = np.where(output.data.cpu().numpy() >= 0, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), pred, average='micro')
        return score

def evaluate_in_batches(dataloader, device, model):
    total_score = 0
    for batch_id, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata['feat']
        labels = batched_graph.ndata['label']
        score = evaluate(batched_graph, features, labels, model)
        total_score += score
    return total_score / (batch_id + 1) # return average score
    
def train(train_dataloader, val_dataloader, test_dataloader, epochs, device, model, fanouts):
    # define loss function and optimizer
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)

    # training loop        
    for epoch in range(epochs):
        model.train()
        logits = []
        total_loss = 0
        # mini-batch loop
        for batch_id, batched_graph in enumerate(train_dataloader):
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata['feat'].float()
            labels = batched_graph.ndata['label'].float()
            seeds = torch.arange(batched_graph.num_nodes())

            blocks, features, labels = convert_to_block(batched_graph, seeds, features, labels, device, fanouts)

            logits = model(blocks, features)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print("Epoch {:05d} | Loss {:.4f} |". format(epoch, total_loss / (batch_id + 1) ), flush=True)
            avg_score = evaluate_in_batches(val_dataloader, device, model) # evaluate F1-score instead of loss
            print("                            Acc. (F1-score) {:.4f} ". format(avg_score))
        if (epoch + 1) % 100 == 0:
            avg_score = evaluate_in_batches(test_dataloader, device, model) # evaluate F1-score instead of loss
            print("                            Acc (Test). (F1-score) {:.4f} ". format(avg_score))
            torch.save(model.state_dict(), f"/home/gwkim/gnn_models/gat_ppi_10_20_30_epoch_{epoch+1}.pt")

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--gnn", type=str, default="gat")

    print(f'Training PPI Dataset with DGL built-in GATConv module.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load and preprocess datasets
    train_dataset = PPIDataset(mode='train')
    val_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    features = train_dataset[0].ndata['feat']
    
    # create GAT model    
    in_size = features.shape[1]
    out_size = train_dataset.num_labels

    if args.gnn == 'gat':
        model = GAT(in_size, 256, out_size, heads=[4,4,6]).to(device)
    else:
        model = SAGE(in_size, 16, out_size).to(device)
    
    # model training
    print('Training...')
    train_dataloader = GraphDataLoader(train_dataset, batch_size=2)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=2)
    fanouts = [int(f) for f in args.fan_out.split(",")]
    train(train_dataloader, val_dataloader, test_dataset, args.epochs, device, model, fanouts)

    # test the model
    print('Testing...')
    test_dataloader = GraphDataLoader(test_dataset, batch_size=2)
    avg_score = evaluate_in_batches(test_dataloader, device, model)
    print("Test Accuracy (F1-score) {:.4f}".format(avg_score))
    torch.save(model.state_dict(), f"/home/gwkim/gnn_models/gat_ppi_{args.gnn}_{args.fan_out.strip()}.pt")
