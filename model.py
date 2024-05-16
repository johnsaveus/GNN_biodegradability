from torch import nn



class GraphAttentionLayer(nn.Module):

# Single attention head

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_heads=1,
                 bias=True):
        
        super(GraphAttentionLayer).__init__()

        self.bias = bias
        self.num_heads = num_heads
        # Transform node embeddings for self attention multiplication
        # Asume Averaging of attention heads
        self.linear_projection = nn.Linear(input_dim, num_heads * output_dim, bias = self.bias) 
        nn.init.xavier_uniform_(self.linear_projection.weight, gain=1.414)
        if self.bias:
            nn.init_zeros_(self.linear_projection.bias)
        # Activation for attention score
        self.activation = nn.LeakyReLU()
        # Normalize attention Scores
        self.softmax = nn.Softmax() 

    def forward(self, graph):






        

    
