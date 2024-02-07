import torch.nn as nn

class V2AMapper(nn.Module):
    def __init__(self, input_dim, output_dim, expansion_rate=4):
        super(V2AMapper, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim * expansion_rate)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(output_dim * expansion_rate)
        self.linear2 = nn.Linear(input_dim * expansion_rate, output_dim)
    
    def forward(self, x):

        x = self.linear(x)
        x = self.silu(x)
        x = self.layer_norm(x)    
        x = self.linear2(x)
        
        return x
