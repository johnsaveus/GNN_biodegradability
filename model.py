from torch import nn
import torch

class GraphAttention(GraphAttentionLayer):

    def __init__(self):
        



class GraphAttentionLayer(nn.Module):

# Single attention head

    def __init__(self,
                 input_feats,
                 output_feats,
                 num_heads=1,
                 drop_prob=0.5,
                 leaky_relu_slope=0.2,
                 concat_heads = False,
                 init_distrib = 'uniform'
                 ):
        
        super(GraphAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.init_distrib = init_distrib
        self.leakyrelu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.softmax = nn.Softmax(dim=-1) 
        self.dropout = nn.Dropout(p=drop_prob)
        # Transform node embeddings for self attention multiplication
        # Asume Averaging of attention heads
        if concat_heads:
            self.output_feats = output_feats
            assert output_feats % num_heads == 0 
            self.num_hidden = output_feats // num_heads
        else:
            self.n_hidden = output_feats

        self.W = nn.Parameter(torch.zeros(size = (input_feats, self.num_hidden * num_heads)))
        # Attention Mechanism
        # Concatination of two nodes connected by an edge (2 * output_dim)
        self.attention = nn.Parameter(torch.zeros(size= (num_heads, 2 * self.num_hidden, 1)))

        self._param_init()

    def _param_init(self):

        if self.init_distrib == 'uniform':
            nn.init.xavier_uniform_(self.W)
            nn.init.xavier_uniform_(self.attention)
        elif self.init_distrib == 'normal':
            nn.init.xavier_normal_(self.W)
            nn.init.xavier_normal_(self.attention)
        else:
            raise ValueError(f"Invalid init_distrib value = {self.init_distrib} Select 'uniform' or 'normal'")
        
    def _calc_attention_scores(self):

        src_scores = 
        target_scores = 
            
    def forward(self, graph):

        node_features, edge_index = graph
        num_nodes = node_features.shape[0]

        assert edge_index.shape[0] == 2, f'Adjacency Matrix needs to be in COO format'

        # Create W*h Matrix. Shape = (Input_dim, Num_heads, Output_Dim)
        linear_proj = torch.mm(node_features, self.W)
        # Dropout to projection
        linear_proj_drop = self.dropout(linear_proj)
        # Index every node and it's neighbor


        #src_projection = linear_projection[edge_index[0]]
        #target_projection = linear_projection[edge_index[1]]
        concat_projections = torch.cat([src_projection, target_projection], dim=-1)
        # Calculate attention_scores per node
        att_scores_raw = self.leakyReLU(torch.matmul(concat_projections, self.attention).squeeze(-1))
        # Normalized
        att_scores_norm = self.softmax(att_scores_raw)
        att_scores_norm = self.dropout(att_scores_norm)

        att_weights = target_projection * att_scores_norm.unsqueeze(dim=-1)

        out = torch.zeros(num_nodes, self.num_heads, self.output_dim).to(att_weights.device)
        out.index_add_(0, edge_index[0], att_weights)

        if self.concat_heads:
            out = out.view(num_nodes, self.num_heads * self.output_dim)  # Concatenate heads
        else:
            out = out.mean(dim=1)  # Average heads

        print(out.shape)
        return out



num_nodes = 5
input_dim = 4
output_dim = 3
num_heads = 2
num_edges = 7

node_features = torch.rand((num_nodes, input_dim))
edge_index = torch.randint(0, num_nodes, (2, num_edges))
graph = (node_features, edge_index)
gat_layer = GraphAttentionLayer(input_dim, output_dim, num_heads)
output = gat_layer(graph)

