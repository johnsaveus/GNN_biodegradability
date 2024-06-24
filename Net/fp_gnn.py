import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
from Net.gat_layer import GraphAttentionLayer
from torch_geometric.nn.pool import global_mean_pool

class FingerprintNN(nn.Module):
    def __init__(self,
                 fp_dim,
                 hidden_feats,
                 drop_prob = 0.2,
                 activation = 'gelu'):
        super(FingerprintNN, self).__init__()
        self.fc1 = nn.Linear(fp_dim, hidden_feats)
        self.dropout = nn.Dropout(drop_prob)
        self.activation = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        self._init_weights()

    def forward(self, fp):
        fp = self.fc1(fp)
        fp = self.activation(fp)
        fp = self.dropout(fp)
        return fp
    
    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)

class GraphAttention_v2(nn.Module):

    def __init__(self,
                 atom_feats,
                 hidden_feats,
                 num_heads,
                 num_layers=3,
                 activation = 'relu',
                 drop_prob=0.2):
        super(GraphAttention_v2, self).__init__() 
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(atom_feats if i == 0 else hidden_feats * num_heads, 
                                hidden_feats, 
                                drop_prob=drop_prob) 
            for i in range(num_layers)
        ])

    def forward(self, x, adj, batch):
        x = F.dropout(x, self.drop_prob, training = self.training)
        #print(x.shape)
        for i, attention_layer in enumerate(self.attention_layers):
            if i == len(self.attention_layers) - 1:
                x = torch.stack([attention_layer(x, adj) for _ in range(self.num_heads)], dim = 0).mean(dim=0)
            else:
                x = torch.cat([attention_layer(x, adj) for _ in range(self.num_heads)], dim=1)
                x = self.activation(x)
                x = F.dropout(x, self.drop_prob, training = self.training)
            #print(x.shape)
        x = self.activation(x)
        x = global_mean_pool(x, batch)
        return x
    

class FP_GNN(nn.Module):

    def __init__(self,
                 hidden_feats,
                 num_heads,
                 num_layers,
                 activation,
                 drop_prob):

        super(FP_GNN, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        self.drop_prob = drop_prob
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.gnn_model = GraphAttention_v2(68,
                                           hidden_feats,
                                           num_heads,
                                           num_layers,
                                           activation,
                                           drop_prob).to(self.device)
        
        self.fpnn_model = FingerprintNN(167,
                                        hidden_feats,
                                        drop_prob,
                                        activation).to(self.device)
        
        self.fc = nn.Linear(hidden_feats * 2, 1)

    def forward(self, x, edge_index, fp, batch):

        gnn_embed = self.gnn_model(x, edge_index, batch)
        fpnn_embed = self.fpnn_model(fp)

        concat_embedd = torch.cat([gnn_embed, fpnn_embed], dim = 1)
        output = self.fc(concat_embedd)
        output = self.activation(output)
        output = F.dropout(output, self.drop_prob, training = self.training)
        output = output.squeeze(dim = -1)
        return output