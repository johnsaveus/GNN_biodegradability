from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from Net.gat_network import GraphAttention
import torch
import json
from torch_geometric.loader import DataLoader
from data.mol_to_graph import MolecularDataset
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_auc_score
import argparse
import os

def create_fold(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_one_epoch(model, train_loader, optimizer, device, base_weight):
    model.train()
    train_loss = 0
    labels = []
    preds = []
    for data in train_loader:     
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch =  data.batch.to(device)
        y_true = data.y.to(device)
        optimizer.zero_grad()
        y_pred = model(x, edge_index, batch)
        weight = base_weight[y_true.long()]
        # data.y is int. Need to be casted to float
        loss_fn = BCEWithLogitsLoss(weight = weight)
        loss = loss_fn(y_pred.to(device), y_true.float())
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        y_probs = torch.sigmoid(y_pred)
        labels.extend(y_true)
        preds.extend((y_probs > 0.5).int())
    roc = roc_auc_score(labels, preds)
    return train_loss, roc

def validate_one_epoch(model, validation_loader, device, base_weight):
    model.eval()
    labels = []
    preds = []
    test_loss = 0
    for data in validation_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch =  data.batch.to(device)
        y_true = data.y.to(device)
        with torch.no_grad():
            y_pred = model(x, edge_index, batch)
            weight = base_weight[y_true.long()]
        # data.y is int. Need to be casted to float
            loss_fn = BCEWithLogitsLoss(weight = weight)
            loss = loss_fn(y_pred.to(device), y_true.float())
        test_loss+=loss.item()
        # data.y is int. Need to be casted to float
        y_probs = torch.sigmoid(y_pred)
        labels.extend(y_true)
        preds.extend((y_probs > 0.5).int())
    roc = roc_auc_score(labels, preds)
    return test_loss, roc

epochs = 100
def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MolecularDataset(root = 'data', 
                            path_to_ind_csv = 'split_ix/csv_train_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_train_ix.txt',
                                save_name='train')
    valid_dataset = MolecularDataset(root = 'data', 
                            path_to_ind_csv = 'split_ix/csv_valid_ix.txt',
                            path_to_ind_sdf = 'split_ix/sdf_valid_ix.txt',
                                save_name='valid')
    
    train_dataset.load("data/processed/train.pt")
    valid_dataset.load("data/processed/valid.pt")
    class_0_weight = 100 / (55 * 2)
    class_1_weight = 100 / (45 * 2)
    base_weight = torch.tensor([class_0_weight, class_1_weight], device = device)
    batch_size = args.batch_size
    train_batches = len(train_dataset) // batch_size
    valid_batches = len(valid_dataset) // batch_size
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader =  DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)
    model = GraphAttention(atom_feats = 68,
                           hidden_feats = args.hidden,
                           num_heads = args.heads,
                           num_layers = args.num_layers,
                           activation = args.activation,
                           dropout_prob= args.dropout_prob,
                           )
    optimizer = AdamW(params = model.parameters(),
                      lr = args.learning_rate,
                      weight_decay = 1e-3)
    train_losses = []
    train_rocs = []
    valid_losses = []
    valid_rocs = []
    for epoch in range(1, epochs+1):
        if epoch % 5 == 0:
            print(f'Epoch = {epoch}')
        train_loss, train_roc = train_one_epoch(model = model,
                                                train_loader = train_loader,
                                                optimizer = optimizer,
                                                device = device,
                                                base_weight = base_weight)
        valid_loss, valid_roc = validate_one_epoch(model = model,
                                                 validation_loader = valid_loader,
                                                 device = device,
                                                 base_weight = base_weight)
        train_losses.append(train_loss / train_batches)
        valid_losses.append(valid_loss / valid_batches)
        train_rocs.append(train_roc)
        valid_rocs.append(valid_roc)

    create_fold('results')
    save_path = 'results/' + args.model_name
    create_fold(save_path)
    args_dict = vars(args)
    args_file = os.path.join(save_path, 'hyperparams.json')
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent = 4)
    print(f"Arguments saved succesfully")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Training and evaluating machine learning model')

    parser.add_argument('--batch-size', type = int, default = 32, help = "Batch size for train and validation loaders")
    parser.add_argument('--hidden', type = int, default = 32, help = "Hidden neuron size for GAT")
    parser.add_argument('--num_heads', type = int, default = 2, help = "Number of attention heads")
    parser.add_argument('--num_layers', type = int, default = 2, help = "Number of Hidden GAT layers")
    parser.add_argument('--activation', type = str, default = 'relu', help = "Activation function for non-linearity")
    parser.add_argument('--drop_prob', type = int, default = 0.1, help = "Dropout probability")
    parser.add_argument('--learning-rate', type = float, default = 0.001, help = "Learning rate")
    parser.add_argument('--model-name', type = str, required = True , help = "Give a name to save")

    args = parser.parse_args()
    main(args)