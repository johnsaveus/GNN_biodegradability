import torch.nn as nn
import torch.nn.init as init
from data.mol_to_graph import MolecularDataset
import torch.nn.functional as F
import torch
from Net.gat_layer import GraphAttentionLayer
from torch_geometric.nn.pool import global_mean_pool
from torch.nn import BCEWithLogitsLoss

class FingerprintNN(nn.Module):
    def __init__(self,
                 fp_dim,
                 drop_prob = 0.2,
                 activation = 'gelu'):
        super(FingerprintNN, self).__init__()
        self.fc1 = nn.Linear(fp_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.dropout = nn.Dropout(drop_prob)
        self.activation = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        self._init_weights()

    def forward(self, fp):

        fp = self.dropout(fp)
        fp = self.fc1(fp)
        fp = self.activation(fp)
        fp = self.dropout(fp)
        fp = self.fc2(fp)
        return fp
    
    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

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
                 drop_gat,
                 drop_fpnn):

        super(FP_GNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.gnn_model = GraphAttention_v2(68,
                                           hidden_feats,
                                           num_heads,
                                           num_layers,
                                           activation,
                                           drop_gat)
        
        self.fpnn_model = FingerprintNN(167,
                                        drop_fpnn,
                                        activation)
        
        self.fc = nn.Linear(hidden_feats * 2, 1)

    def forward(self, x, edge_index, fp, batch):

        gnn_embed = self.gnn_model(x, edge_index, batch).to(self.device)
        fpnn_embed = self.fpnn_model(fp).to(self.device)

        concat_embedd = torch.cat([gnn_embed, fpnn_embed], dim = 1)

        output = self.fc(concat_embedd)
        output = output.squeeze(dim = -1)
        return output
    
# train_dataset = MolecularDataset(root = '../data', 
#                             path_to_ind_csv = 'split_ix/csv_train_ix.txt',
#                             path_to_ind_sdf = 'split_ix/sdf_train_ix.txt',
#                             save_name='train')

# data = train_dataset[0]
# fp_gnn = FP_GNN(16, 4, 2, 'gelu', 0.2, 0.2)

# loss = BCEWithLogitsLoss()
# output = fp_gnn(data.x, data.edge_index, data.fingerprint, data.batch)

# loss_fn = loss(output, data.y.float())


# loss_fn.backward()






        



# train_dataset = MolecularDataset(root = '../data', 
#                             path_to_ind_csv = 'split_ix/csv_train_ix.txt',
#                             path_to_ind_sdf = 'split_ix/sdf_train_ix.txt',
#                                 save_name='train')
# valid_dataset = MolecularDataset(root = '../data', 
#                             path_to_ind_csv = 'split_ix/csv_valid_ix.txt',
#                             path_to_ind_sdf = 'split_ix/sdf_valid_ix.txt',
#                                 save_name='valid')
# train_dataset.load('../data/processed/train.pt')

# data = train_dataset[0]
# from torch.optim import AdamW
# torch.manual_seed(42)
# model_gat = GraphAttention_v2(68, 16, 4, 2)
# model_fpnn = FingerprintNN(167, 0.1)
# optimizer_1 = AdamW(params = model_fpnn.parameters())
# optimizer_2 = AdamW(params = model_gat.parameters())
# for epoch in range(100):
#     model_gat.train()
#     model_fpnn.train()
#     y1 = model_gat(data.x, data.edge_index, data.batch)
#     y2 = model_fpnn(data.fingerprint)
#     # print(y1.shape)
#     # print(y2.shape)

#     y = torch.concat([y1, y2], dim = 1)
#     # print(y.shape)
#     fc = nn.Linear(32, 1)
#     y_out = fc(y).squeeze(dim = 0)
#     # print(y_out.shape)
#     # print(y_out)
#     optimizer_1.zero_grad()
#     optimizer_2.zero_grad()
#     loss_fn = BCEWithLogitsLoss()
#     loss = loss_fn(y_out, data.y.float())
#     loss.backward()
#     optimizer_1.step()
#     optimizer_2.step()
#     print(loss)

    
        