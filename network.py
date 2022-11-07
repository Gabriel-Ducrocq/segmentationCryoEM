import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np
from mlp import MLP


class Net(torch.nn.Module):
    def __init__(self, N_residues, N_domains, B, S, decoder, mlp_translation, local_frame, atom_absolute_positions):
        super(Net, self).__init__()
        self.N_residues = N_residues
        self.N_domains = N_domains
        self.B = B
        self.S = S
        self.decoder = decoder
        self.mlp_translation = mlp_translation
        self.local_frame = torch.transpose(local_frame, 0, 1)
        self.atom_absolute_positions = atom_absolute_positions

        nb_per_res = int(B / S)
        balance = B - S + 1
        self.bs_per_res = np.empty((self.N_residues, nb_per_res), dtype=int)
        for i in range(self.N_residues):
            start = max(i + balance - B, 0) // S  ##Find the smallest window number such that the residue i is inside

            if max(i + balance - B, 0) % S != 0:
                start += 1  ## If there is a rest, means we have to go one more window

            self.bs_per_res[i, :] = np.arange(start, start + nb_per_res)

        nb_windows = self.bs_per_res[-1, -1] + 1
        #alpha = torch.randn((nb_windows, self.N_domains))
        alpha = torch.ones((nb_windows, self.N_domains))
        self.weights = torch.nn.Parameter(data=alpha, requires_grad=True)

    def multiply_windows_weights(self):
        weights_per_residues = torch.empty((self.N_residues, self.N_domains))
        for i in range(self.N_residues):
            windows_set = self.bs_per_res[i]  # Extracting the indexes of the windows for the given residue
            weights_per_residues[i, :] = torch.prod(self.weights[windows_set, :],
                                                    axis=0)  # Muliplying the weights of all the windows, for each subsystem

        attention_softmax = F.softmax(weights_per_residues, dim=1)
        return attention_softmax

    def func(self):
        attention_softmax = self.multiply_windows_weights()
        attention_softmax_log = torch.log(attention_softmax)
        prod = attention_softmax_log * attention_softmax
        loss = -torch.sum(prod)

        return loss / self.N_residues

    def deform_structure(self, weights, translation_scalars):
        ##Modifying to apply translation directly
        translation_vectors = torch.matmul(translation_scalars, self.local_frame)
        translation_per_residue = torch.matmul(weights, translation_vectors)
        #translation_per_residue = translation_vectors
        #new_atom_positions = self.atom_absolute_positions + torch.repeat_interleave(translation_per_residue, 3, 0)
        new_atom_positions = self.atom_absolute_positions + torch.repeat_interleave(translation_per_residue, 3, 0)
        return new_atom_positions, translation_per_residue

    def forward(self, x, edge_index, edge_attr, latent_variables):
        weights = self.multiply_windows_weights()
        #hidden_representations = self.decoder.forward(x, edge_index, edge_attr, latent_variables)
        #weights_transp = torch.transpose(weights, 0, 1)
        #features_domain = torch.matmul(weights_transp, hidden_representations)
        #features_domain = torch.sum(hidden_representations, dim = 0)
        #features_and_latent = torch.concat([features_domain, torch.broadcast_to(latent_variables,(1, 3))], dim=1)
        #features_and_latent = torch.concat([torch.broadcast_to(features_domain, (1, 50)), torch.broadcast_to(latent_variables, (1,3))], dim=1)
        features_and_latent = torch.concat([torch.randn((2, 50)), torch.reshape(latent_variables, (2, 3))], dim=1)
        #features_and_latent = torch.broadcast_to(latent_variables, (1, 3))
        scalars_per_domain = self.mlp_translation.forward(features_and_latent)
        #print("Test:", features_and_latent)
        ##Tweaking to get only a translation vector
        new_structure, translations = self.deform_structure(weights, scalars_per_domain)
        return new_structure, weights, translations


    def loss(self, new_structure, true_structure, mask_weights):
    #def loss(self, new_translation, true_translation, mask_weights):
        rmsd = torch.sqrt(torch.mean(torch.sum((new_structure - true_structure)**2, dim=1)))
        #rmsd = torch.sum((new_translation - true_translation)**2)
        attention_softmax_log = torch.log(mask_weights)
        prod = attention_softmax_log * mask_weights
        loss = -torch.sum(prod)

        return loss / self.N_residues + 3*rmsd



