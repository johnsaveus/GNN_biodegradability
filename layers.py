import torch
import torch.nn as nn
import torch.nn.functional as F
from featurizer import MolecularDataset

class GraphAttentionLayer(nn.Module):

# Single attention head

    def __init__(self,
                 in_feats,
                 out_feats,
                 #num_heads=1,
                 drop_prob=0.5,
                 leaky_relu_slope=0.2,
                 #concat_heads = False,
                 init_distrib = 'uniform'
                 ):
        
        super(GraphAttentionLayer, self).__init__()

        #self.num_heads = num_heads
        self.out_feats = out_feats
        #self.concat_heads = concat_heads
        self.init_distrib = init_distrib
        self.leakyrelu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout = nn.Dropout(p=drop_prob)
        # Weight matrix for the linear projection of node features W*h. 
        # This is for a single attention head
        self.W = nn.Parameter(torch.empty(size = (in_feats,out_feats)))
        # Attention weight vector.
        self.a = nn.Parameter(torch.empty(size= (2 * out_feats, 1)))
        self._params_init()

    def _params_init(self):
        g = 1.414
        if self.init_distrib == 'uniform':
            nn.init.xavier_uniform_(self.W, gain=g)
            nn.init.xavier_uniform_(self.a, gain=g)
        elif self.init_distrib == 'normal':
            nn.init.xavier_normal_(self.W, gain=g)
            nn.init.xavier_normal_(self.a, gain=g)
        else:
            raise ValueError(f"Invalid init_distrib value = {self.init_distrib} Select 'uniform' or 'normal'")
        
    def _calc_attention_scores(self, linear_proj):
        # Broadcasting
        Wh_i = torch.matmul(linear_proj, self.a[:self.out_feats, :])
        Wh_j = torch.matmul(linear_proj, self.a[self.out_feats:, :])
        # Broadcast operation instead of concat.
        Wh = Wh_i + Wh_j.mT
        e_i_j = self.leakyrelu(Wh)
        return e_i_j
    
    def _convert_adj_to_dense(self, x, adj):

        num_nodes = x.shape[0]
        vals = torch.ones(adj.shape[1], dtype = torch.float32)
        sparse_adj = torch.sparse_coo_tensor(adj, vals, (num_nodes, num_nodes))
        dense_adj = sparse_adj.to_dense()
        bool_dense = dense_adj > 0
        return bool_dense
    def forward(self, x, adj):

        #if adj.shape[0]!= x.shape[0] and adj.shape[1] != x.shape[1]:
        adj = self._convert_adj_to_dense(x, adj)
        #num_nodes = node_features.shape[0]
        #assert edge_index.shape[0] == 2, f'Adjacency Matrix needs to be in COO format'
        # Create W*h Matrix with heads.
        # (num_nodes , in_feats) * (in_feats, out_feats).  Shape = (Num_nodes, output_feats)
        # Multiplication without broadcasting
        #print(x.shape)
        linear_proj = torch.mm(x, self.W)
        # Dropout to projection
        #print(linear_proj.shape)
        linear_proj_drop = self.dropout(linear_proj)
        # Calculate e before normalizing. 
        e = self._calc_attention_scores(linear_proj_drop)
        # Only the connected nodes contribute to attention
        # Init zero_vec with low values so it can be near zero when using softmax
        softmax_zero_vals = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0 , e, softmax_zero_vals)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        #print(attention.shape)
        # Final multiplication for the input of the next layer.
        # Reminder: 1 head
        h_out = torch.matmul(attention, linear_proj)

        return h_out
       # if self.concat_heads:
       #     return F.elu(h_out)
       # else:
       #     return h_out




#dataset = MolecularDataset(root = 'data')
#dataset.load('data\processed\data.pt')
#gat_layer = GraphAttentionLayer(60, 40)
#output = gat_layer(dataset[0].x, dataset[0].edge_index)
#print(output.shape)
# Seems like it works
