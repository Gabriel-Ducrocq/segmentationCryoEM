import torch
#import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt



class Renderer():
    def __init__(self, pixels_x, pixels_y, N_heavy=4530, std = 1):
        self.std_blob = std
        self.pixels_x = torch.tensor(pixels_x, dtype=torch.float32)
        self.pixels_y = torch.tensor(pixels_y, dtype=torch.float32)
        self.N_heavy_atoms = N_heavy
        self.torch_sqrt_2pi= torch.sqrt(torch.tensor(2*np.pi))

    def compute_gaussian_kernel(self, x, pixels_pos):
        """
        Computes the values of the gaussian kernel for one axis only but all heavy atoms and samples in batch
        :param x: (N_batch, 1): the coordinate of all heavy atoms on one axis for all samples in batch.
        :return: (N_batch, N_atoms, N_pix)
        """
        batch_size = x.shape[0]
        scaled_distances = -(1/2)*(torch.broadcast_to(pixels_pos, (batch_size, self.N_heavy_atoms, -1)) -
                                   x[:, :, None])**2/self.std_blob
        axis_val = torch.exp(scaled_distances)/self.torch_sqrt_2pi
        return axis_val

    def compute_x_y_values_all_atoms(self, atom_positions):
        """

        :param atom_position: (N_batch, N_atoms, 3)
        :return:
        """
        all_x = self.compute_gaussian_kernel(atom_positions[:, :, 0], self.pixels_x)
        all_y = self.compute_gaussian_kernel(atom_positions[:, :, 1], self.pixels_y)

        prod= torch.einsum("bki,bkj->bkij", (all_x, all_y))
        #prod = torch.bmm(all_x, all_y)
        projected_densities = torch.sum(prod, dim=1)
        #print("Proj:", projected_densities.shape)
        #fft_densities = torch.fft.rfft2(projected_densities)
        #print(fft_densities.shape)
        #CTF_one_dim = torch.sin(torch.linspace(0, 50, 50))
        #CTF = torch.broadcast_to(torch.outer(CTF_one_dim, CTF_one_dim), (1, 50, 50))
        #print("CTF",CTF.shape)
        #print(fft_densities.shape)
        #torch.einsum("bki,bkj->bkij", (all_x, all_y))
        #fft_densities = CTF*fft_densities
        #projected_densities = torch.fft.irfft2(fft_densities)
        ### ADD THE CTF !!!
        return projected_densities #+ torch.randn(size=(1, 256, 256))*0.5



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

pixels_x = np.linspace(-70, 70, num = 256).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num = 256).reshape(1, -1)
rend = Renderer(pixels_x, pixels_y, std=1)
print(torch.min(absolute_positions[0, :, 0]))
print(torch.max(absolute_positions[0, :, 0]))

print(torch.min(absolute_positions[0, :, 1]))
print(torch.max(absolute_positions[0, :, 1]))

res = rend.compute_x_y_values_all_atoms(absolute_positions)
res = res[0].detach().numpy()

print(np.unique(res))
print(res.shape)

plt.imshow(res, cmap="gray")
plt.show()
"""


