import torch
import torch.nn as nn
import torch.nn.functional as F
from Net.gat_layer import GraphAttentionLayer
from torch_geometric.nn.pool import global_mean_pool

class GraphAttention(nn.Module):

    def __init__(self,
                 atom_feats,
                 hidden_feats,
                 num_heads,
                 num_layers=3,
                 activation = 'relu',
                 drop_prob=0.2):
        super(GraphAttention, self).__init__() 
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        self.fc1 = nn.Linear(hidden_feats, 1)
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(atom_feats if i == 0 else hidden_feats * num_heads, 
                                hidden_feats, 
                                drop_prob=drop_prob,
                                init_distrib = 'uniform') 
            for i in range(num_layers)
        ])

    def forward(self, x, adj, batch):
        x = F.dropout(x, self.drop_prob, training = self.training)
        for i, attention_layer in enumerate(self.attention_layers):
            if i == len(self.attention_layers) - 1:
                x = torch.stack([attention_layer(x, adj) for _ in range(self.num_heads)], dim = 0).mean(dim=0)
            else:
                x = torch.cat([attention_layer(x, adj) for _ in range(self.num_heads)], dim=1)
                x = self.activation(x)
                x = F.dropout(x, self.drop_prob, training = self.training)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = x.squeeze(dim = -1)
        return x