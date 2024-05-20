import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class GraphAttention(nn.Module):

    def __init__(self,
                 atom_feats,
                 hidden_feats,
                 num_heads,
                 drop_prob=0.5):
        super(GraphAttention, self).__init__()
        
        self.drop_prob = drop_prob
        self.multiple_heads = [GraphAttentionLayer(atom_feats, 
                                                  hidden_feats,
                                                  drop_prob = self.drop_prob, 
                                                  concat_heads = True)
                               for _ in range(num_heads)]
        
        for layer_ix, attention in enumerate(self.multiple_heads):
            self.add_module(f'Layer {layer_ix}, {attention}')

        self.avg_heads = GraphAttentionLayer(hidden_feats * num_heads,
                                             1, # classification
                                            dropout = self.drop_prob,
                                            concat_heads = False)
        
    def forward(self, x, adj):

        x = F.dropout(x, self.drop_prob, training = self.training)
        x = torch.cat([head(x, adj) for head in self.multiple_heads], dim = 1)
        x = F.dropout(x, self.drop_prob, training = self.training)
        x = F.elu(self.avg_heads(x, adj))
        return F.log_softmax(x, dim=1)
