import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, intermediate_dim, num_hidden_layers = 1):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.num_hidden_layers = num_hidden_layers
        ##Using leaky RELU !
        self.input_layer = nn.Sequential(nn.Linear(in_dim, intermediate_dim), nn.LeakyReLU())
        self.output_layer = nn.Sequential(nn.Linear(intermediate_dim, out_dim))
        list_intermediate = [nn.Sequential(nn.Linear(intermediate_dim, intermediate_dim), nn.LeakyReLU())
                             for _ in range(num_hidden_layers)]
        self.linear_relu_stack = nn.Sequential(*[layer for layer in list_intermediate])

    def forward(self, x):
        #x = self.flatten(x)
        x = self.input_layer(x)
        hidden = self.linear_relu_stack(x)
        logits = self.output_layer(hidden)
        return logits
