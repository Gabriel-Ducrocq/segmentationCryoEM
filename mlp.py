import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, intermediate_dim, device, num_hidden_layers = 1, network_type="decoder"):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.type=network_type
        self.out_dim = out_dim
        self.output_ELU = torch.nn.ELU()
        if type(intermediate_dim) == type([]):
            self.num_hidden_layers = len(intermediate_dim)
            self.input_layer = nn.Sequential(nn.Linear(in_dim, intermediate_dim[0], device=device), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(intermediate_dim[-1], out_dim, device=device))
            list_intermediate = [nn.Sequential(nn.Linear(intermediate_dim[i], intermediate_dim[i+1], device=device), nn.LeakyReLU())
                             for i in range(self.num_hidden_layers-1)]
            self.linear_relu_stack = nn.Sequential(*[layer for layer in list_intermediate])
        else:
            self.input_layer = nn.Sequential(nn.Linear(in_dim, intermediate_dim, device=device), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(intermediate_dim, out_dim, device=device))
            list_intermediate = [nn.Sequential(nn.Linear(intermediate_dim, intermediate_dim, device=device), nn.LeakyReLU())
                                 for _ in range(num_hidden_layers)]
            self.linear_relu_stack = nn.Sequential(*[layer for layer in list_intermediate])

    def forward(self, x):
        #x = self.flatten(x)
        x = self.input_layer(x)
        hidden = self.linear_relu_stack(x)
        output = self.output_layer(hidden)
        if self.type == "encoder":
            latent_mean = output[:, :int(self.out_dim/2)]
            latent_std = self.output_ELU(output[:, int(self.out_dim/2):]) + 1
            output_with_std = torch.cat([latent_mean, latent_std], dim=-1)
            print("OUTPUT STD", output_with_std)
            return output_with_std

        return output
