from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam
from models import GraphAttention
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from featurizer import MolecularDataset
#from torch.utils.data import DataLoader

loss_fn  = BCEWithLogitsLoss()
epochs = 50

dataset = MolecularDataset(root = 'data')
dataset.load('data\processed\data.pt')
train_data = dataset[1000:1500]
test_data  = dataset[5000:6000]
#for data in train_data:
 #   print(data)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
model = GraphAttention(60, 20, 4)
optimizer = Adam(params = model.parameters(), lr = 0.001)

for epoch in range(epochs):
    total_loss = 0
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
        total_loss+= loss
    print(total_loss/32)
model.eval()
y_preds = []
y_true = []
for data in test_data:
    #print(data)
    x, edge_index, y, batch = data.x, data.edge_index, data.batch, data.batch
    y_pred = model(x, edge_index, batch)
    #print(y_pred.item())
    y_preds.append(y_pred.item())
    y_true.append(data.y.tolist())
    #print(data['y'].tolist())
#print(y_true)
y_preds = [1 if pred>=0 else 0 for pred in y_preds]

#print(y_preds)
print(confusion_matrix(y_true, y_preds))
