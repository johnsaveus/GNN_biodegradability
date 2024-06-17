import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import global_mean_pool
from data.mol_to_graph import MolecularDataset
import torch

class GraphTransformer(nn.Module):

    def __init__(self,
                 atom_feats,
                 hidden_feats,
                 num_heads,
                 num_layers=3,
                 activation = 'relu',
                 drop_prob=0.2):
        super(GraphTransformer, self).__init__()
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        self._select_activation(activation)
        self.transformer_layers = nn.ModuleList([
            TransformerConv(atom_feats if i == 0 else hidden_feats * num_heads, 
                            out_channels = hidden_feats, 
                            heads = num_heads,
                            dropout = drop_prob,
                            concat = True)
            for i in range(num_layers-1)])

        self.final_transformer = TransformerConv(hidden_feats * num_heads,
                                                 out_channels = hidden_feats,
                                                 heads = num_heads,
                                                 dropout = drop_prob,
                                                 concat = False)
        self.fc = nn.Linear(hidden_feats, 1)
    
    def forward(self, x, edge_index, batch):
        edge_index = edge_index.to(torch.int64)
        for layer in self.transformer_layers:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, self.drop_prob, training = self.training)
        
        x = self.final_transformer(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        x = x.squeeze(dim = -1)
        return x

    def _select_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()

# train_dataset = MolecularDataset(root = '../data', 
#                             path_to_ind_csv = 'split_ix/csv_train_ix.txt',
#                             path_to_ind_sdf = 'split_ix/sdf_train_ix.txt',
#                                 save_name='train')
# train_dataset.load('../data/processed/train.pt')

# data = train_dataset[0]

# torch.manual_seed(42)
# model = GraphTransformer(68, 2, 2, 2)
# y = model(data.x, data.edge_index, data.batch)
# print(y)