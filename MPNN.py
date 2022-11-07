import torch
import numpy as np
import torch_geometric
from torch import nn
from torch_geometric.nn import MessagePassing


class MessagePassingNetwork(MessagePassing):
    def __init__(self, message_mlp, update_mlp, num_nodes, num_edges, latent_dim=3):
        super().__init__(aggr="add", flow="source_to_target")
        self.message_mlp = message_mlp
        self.update_mlp = update_mlp
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def forward(self, x, edge_index, edge_attr, latent_variables):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, latent_variables=latent_variables)
        return out

    def message(self, x_i, x_j, edge_attr, latent_variables):
        latent_variables = torch.broadcast_to(latent_variables, (self.num_edges, self.latent_dim))
        x = torch.cat((x_i, x_j, edge_attr, latent_variables), dim = 1)
        return self.message_mlp.forward(x)

    def update(self, aggregated_i, x, latent_variables):
        latent_variables = torch.broadcast_to(latent_variables, (self.num_nodes, self.latent_dim))
        x = torch.cat((x, aggregated_i, latent_variables), dim = 1)
        return self.update_mlp.forward(x)
