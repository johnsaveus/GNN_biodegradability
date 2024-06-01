import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer
from torch_geometric.nn.pool import global_add_pool
from featurizer import MolecularDataset

torch.manual_seed(42)
class GraphAttention(nn.Module):

    def __init__(self,
                 atom_feats,
                 hidden_feats,
                 num_heads,
                 num_layers=1,
                 activation = nn.LogSigmoid(),
                 drop_prob=0.5):
        super(GraphAttention, self).__init__()
        
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        self.activation = activation
        self.fc1 = nn.Linear(hidden_feats * num_heads, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, 1)
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(atom_feats if i == 0 else hidden_feats * num_heads, 
                                hidden_feats, 
                                drop_prob=drop_prob) 
            for i in range(num_layers)
        ])

    def forward(self, x, adj, batch):
        #print(x.shape)
        x = F.dropout(x, self.drop_prob, training = self.training)
        #print(x.shape)
        for attention_layer in self.attention_layers:
            x = torch.cat([attention_layer(x, adj) for _ in range(self.num_heads)], dim = 1)
            #print(x.shape)
        x = self.activation(x)
        x = F.dropout(x, self.drop_prob, training = self.training)
        #x = x.sum(dim=1) / x.shape[0]
        x = global_add_pool(x, batch)
        #print(x.shape)
        #print(x.shape)
       # print(x.shape)
        x = self.fc1(x)
       # print(x.shape)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
       # print(x)
        #x = x.squeeze(dim = -1)
        #print(x.shape)
        x = x.squeeze(dim = -1)
        return x
    
class FPNN(nn.Module):

    def __init__(self,
                 fp_dim,
                 fc1_dim,
                 fc2_dim,
                 activation = nn.ReLU(),
                 drop_prob=0.5):
        
        super(FPNN, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.activation = activation
        self.fc1 = nn.Linear(fp_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, 1)

    def forward(self, fingerprint):

        out = self.fc1(fingerprint)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = out.squeeze(dim = -1)

        return out

dataset = MolecularDataset(root = 'data')
dataset.load('data\processed\data.pt')
fp_size = dataset[0].fingerprint.shape[0]
fp_network = FPNN(fp_size, 500, 250)
output = fp_network(dataset[0].fingerprint)
print(output)
'''
dataset = MolecularDataset(root = 'data')
dataset.load('data\processed\data.pt')
gat_layer = GraphAttention(60, 40, 4)
output = gat_layer(dataset[0].x, dataset[0].edge_index)
print(output)
'''