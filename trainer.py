from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam
from models import GraphAttention, FPNN
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from featurizer import MolecularDataset
from torch_geometric.nn.models import GAT
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

loss_fn  = BCEWithLogitsLoss()
epochs = 50

dataset_train = MolecularDataset(root = 'data', path_to_ind='data/split_ix/train_ix.txt', save_name='train')
dataset_valid = MolecularDataset(root = 'data', path_to_ind='data/split_ix/valid_ix.txt', save_name='valid')
dataset_test = MolecularDataset(root = 'data', path_to_ind='data/split_ix/test_ix.txt', save_name='test')
dataset_train.load('data/processed/train.pt')
dataset_valid.load('data/processed/valid.pt')
dataset_test.load('data/processed/test.pt')

print(dataset_train[0])
train_batches = len(dataset_train) / 64
valid_batches = len(dataset_valid) / 64

train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
validation_loader = DataLoader(dataset=dataset_valid, batch_size=64, shuffle=False)

model = GraphAttention(60, 30, 4)
optimizer = Adam(params = model.parameters(), lr = 0.001)

for epoch in range(epochs):
    model.train()
    print(epoch)
    train_loss = 0
    test_loss = 0
    for data in train_loader:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        torch.manual_seed(42)
        y_pred = model(x, edge_index, batch)
        # data.y is int. Need to be casted to float
        loss = loss_fn(y_pred, data.y.float())
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    train_loss = train_loss / train_batches
    print('TRAINING ----')
    print(train_loss)
    model.eval()
    with torch.no_grad():
        for data in validation_loader:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            torch.manual_seed(42)
            y_pred = model(x, edge_index, batch)
            # data.y is int. Need to be casted to float
            loss = loss_fn(y_pred, data.y.float())
            test_loss+= loss.item()
    test_loss = test_loss / valid_batches
    print('VAL------')
    print(test_loss)