import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 drop_prob=0.5,
                 leaky_relu_slope=0.2,
                 init_distrib = 'uniform'):
        
        super(GraphAttentionLayer, self).__init__()
        self.out_feats = out_feats
        self.init_distrib = init_distrib
        self.leakyrelu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout = nn.Dropout(p=drop_prob)
        # This is for a single attention head
        self.W = nn.Parameter(torch.empty(size = (in_feats,out_feats))) # Used for W * h 
        # Attention weight vector
        self.a = nn.Parameter(torch.empty(size= (2 * out_feats, 1)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self._params_init()

    def _params_init(self):
        g = 1.414
        if self.init_distrib == 'uniform':
            nn.init.xavier_uniform_(self.W, gain=g)
            nn.init.xavier_uniform_(self.a, gain=g)
        elif self.init_distrib == 'normal':
            nn.init.xavier_normal_(self.W, gain=g)
            nn.init.xavier_normal_(self.a, gain=g)
        elif self.init_distrib == 'kaiming':
            nn.init.kaiming_normal_(self.W, nonlinearity='leaky_relu', a=self.leakyrelu.negative_slope)
            nn.init.kaiming_normal_(self.a, nonlinearity='leaky_relu', a=self.leakyrelu.negative_slope)
        else:
            raise ValueError(f"Invalid init_distrib value = {self.init_distrib} Select 'uniform' or 'normal'")
        
    def _calc_attention_scores(self, linear_proj):
        # Broadcasting
        Wh_i = torch.matmul(linear_proj, self.a[:self.out_feats, :].to(self.device))
        Wh_j = torch.matmul(linear_proj, self.a[self.out_feats:, :].to(self.device))
        # Broadcast operation instead of concat.
        Wh = Wh_i + Wh_j.mT
        e_i_j = self.leakyrelu(Wh)
        return e_i_j
    
    def _convert_adj_to_dense(self, x, adj):
        device = x.device
        num_nodes = x.shape[0]
        vals = torch.ones(adj.shape[1], dtype = torch.float32, device = device)
        sparse_adj = torch.sparse_coo_tensor(adj, vals, (num_nodes, num_nodes), device = device)
        dense_adj = sparse_adj.to_dense()
        bool_dense = dense_adj > 0
        return bool_dense
    
    def forward(self, x, adj):
        device = x.device
        adj = self._convert_adj_to_dense(x, adj).to(device)
        # Create W*h Matrix.
        # (num_nodes , in_feats) * (in_feats, out_feats).  Shape = (Num_nodes, output_feats)
        # Multiplication without broadcasting
        linear_proj = torch.mm(x, self.W.to(device))
        # Calculate e before normalizing. 
        e = self._calc_attention_scores(linear_proj)
        # Only the connected nodes contribute to attention
        # Init zero_vec with low values so it can be near zero when using softmax
        softmax_zero_vals = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0 , e, softmax_zero_vals).to(device)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        # Final multiplication for the input of the next layer.
        # Reminder: 1 head
        h_out = torch.matmul(attention.to(device), linear_proj.to(device))
        return h_out