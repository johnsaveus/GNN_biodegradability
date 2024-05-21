import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer
from torch_geometric.nn.pool import global_add_pool

torch.manual_seed(42)
class GraphAttention(nn.Module):

    def __init__(self,
                 atom_feats,
                 hidden_feats,
                 num_heads,
                 num_layers=1,
                 drop_prob=0.5):
        super(GraphAttention, self).__init__()
        
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        self.fc1 = nn.Linear(hidden_feats * num_heads, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, 1)
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(atom_feats if i == 0 else hidden_feats * num_heads, 
                                hidden_feats, 
                                drop_prob=drop_prob, 
                                concat_heads=True) 
            for i in range(num_layers)
        ])

    def forward(self, x, adj, batch):

        x = F.dropout(x, self.drop_prob, training = self.training)
        for attention_layer in self.attention_layers:
            x = torch.cat([attention_layer(x, adj) for _ in range(self.num_heads)], dim = 1)
            #print(x.shape)
        x = F.dropout(x, self.drop_prob, training = self.training)
        x = global_add_pool(x, batch)
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x)
        #print(x.shape)
        x = torch.sigmoid(x)
        print(x)
        return x

dataset = torch.load('data/train.pt')
data = dataset[0]
graph = (data.x, data.edge_index, data.batch)
gat_layer = GraphAttention(60, 40, 4)
output = gat_layer(data.x, data.edge_index, data.batch)