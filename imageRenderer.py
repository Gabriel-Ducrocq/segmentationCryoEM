import torch
#import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class Renderer():
    def __init__(self, pixels_x, pixels_y, N_heavy=4530, std = 1, device="cpu"):
        self.std_blob = std
        self.len_x = pixels_x.shape[1]
        self.len_y = pixels_y.shape[1]
        self.pixels_x = torch.tensor(pixels_x, dtype=torch.float32, device=device)
        self.pixels_y = torch.tensor(pixels_y, dtype=torch.float32, device=device)
        self.grid_x = self.pixels_x.repeat(self.len_y, 1)
        self.grid_y = torch.transpose(self.pixels_y, dim0=-2, dim1=-1).repeat(1, self.len_x)
        self.grid = torch.concat([self.grid_x[:, :, None], self.grid_y[:, :, None]], dim=2)
        self.N_heavy_atoms = N_heavy
        self.torch_sqrt_2pi= torch.sqrt(torch.tensor(2*np.pi, device=device))

    def compute_gaussian_kernel(self, x, pixels_pos):
        """
        Computes the values of the gaussian kernel for one axis only but all heavy atoms and samples in batch
        :param x: (N_batch, 1): the coordinate of all heavy atoms on one axis for all samples in batch.
        :return: (N_batch, N_atoms, N_pix)
        """
        batch_size = x.shape[0]
        scaled_distances = -(1/2)*(torch.broadcast_to(pixels_pos, (batch_size, self.N_heavy_atoms, -1)) -
                                   x[:, :, None])**2/self.std_blob**2
        axis_val = torch.exp(scaled_distances)/self.torch_sqrt_2pi
        return axis_val

    #def compute_gaussian_kernel(self, x):
    #    """
    #    Computes the values of the gaussian kernel for one axis only but all heavy atoms and samples in batch
    #    :param x: (N_batch, 1): the coordinate of all heavy atoms on one axis for all samples in batch.
    #    :return: (N_batch, N_atoms, N_pix)
    #    """
    #    batch_size = x.shape[0]
    #    expended_atom_positions = torch.broadcast_to(x[:, :, None, None, :], (batch_size, self.N_heavy_atoms, self.len_y, self.len_x, 2))
    #    exponent = torch.exp(-0.5*torch.sum((expended_atom_positions - self.grid)**2/self.std_blob**2, dim=-1))/(self.torch_sqrt_2pi**2*
    #    self.std_blob**2)
    #    return torch.sum(exponent, dim=1)

    def compute_x_y_values_all_atoms(self, atom_positions, rotation_matrices):
        """

        :param atom_position: (N_batch, N_atoms, 3)
        :rotation_matrices: (N_batch, 3, 3)
        :return:
        """
        transposed_atom_positions = torch.transpose(atom_positions, dim0=1, dim1=2)
        rotated_transposed_atom_positions = torch.matmul(rotation_matrices, transposed_atom_positions)
        rotated_atom_positions = torch.transpose(rotated_transposed_atom_positions, 1, 2)
        #projected_densities = self.compute_gaussian_kernel(rotated_atom_positions[:, :, :2])
        all_x = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 0], self.pixels_x)
        all_y = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 1], self.pixels_y)
        prod= torch.einsum("bki,bkj->bkij", (all_x, all_y))
        projected_densities = torch.sum(prod, dim=1)
        return projected_densities



"""
file = "data/features.npy"
device = "cpu"
features = np.load(file, allow_pickle=True)
features = features.item()
absolute_positions = features["absolute_positions"]
absolute_positions = absolute_positions - np.mean(absolute_positions, axis=0)
absolute_positions = absolute_positions.reshape(1, -1, 3)
absolute_positions = torch.tensor(absolute_positions).to(device)
print(absolute_positions.shape)

pixels_x = np.linspace(-150, 150, num = 128).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num = 128).reshape(1, -1)
rend = Renderer(pixels_x, pixels_y, std=1)
print(torch.min(absolute_positions[0, :, 0]))
print(torch.max(absolute_positions[0, :, 0]))

print(torch.min(absolute_positions[0, :, 1]))
print(torch.max(absolute_positions[0, :, 1]))

res = rend.compute_x_y_values_all_atoms(absolute_positions, torch.eye(3)[None, :, :])
res = res[0].detach().numpy()

print(np.unique(res))
print(res.shape)

plt.imshow(res.T, cmap="gray")
plt.show()
"""


