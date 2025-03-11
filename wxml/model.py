import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()

        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)