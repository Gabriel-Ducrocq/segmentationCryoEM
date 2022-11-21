import torch
#import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np
from mlp import MLP
from torch.utils.tensorboard import SummaryWriter


class Net(torch.nn.Module):
    def __init__(self, N_residues, N_domains, B, S, decoder, mlp_translation, local_frame, atom_absolute_positions,
                 batch_size, cutoff1, cutoff2, device, alpha_entropy = 0.0001):
        super(Net, self).__init__()
        self.N_residues = N_residues
        self.N_domains = N_domains
        self.B = B
        self.S = S
        self.epsilon_mask_loss = 1e-10
        self.decoder = decoder
        self.mlp_translation = mlp_translation
        self.local_frame = torch.transpose(local_frame, 0, 1)
        self.atom_absolute_positions = atom_absolute_positions
        self.batch_size = batch_size
        self.alpha_entropy = alpha_entropy
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.device = device

        nb_per_res = int(B / S)
        balance = B - S + 1
        self.bs_per_res = np.empty((self.N_residues, nb_per_res), dtype=int)
        self.bs_per_res = torch.tensor(self.bs_per_res, device=device)
        for i in range(self.N_residues):
            start = max(i + balance - B, 0) // S  ##Find the smallest window number such that the residue i is inside

            if max(i + balance - B, 0) % S != 0:
                start += 1  ## If there is a rest, means we have to go one more window

            self.bs_per_res[i, :] = torch.arange(start, start + nb_per_res, device=device)

        nb_windows = self.bs_per_res[-1, -1] + 1
        #alpha = torch.randn((nb_windows, self.N_domains))
        alpha = torch.ones((nb_windows, self.N_domains), device=device)
        self.weights = torch.nn.Parameter(data=alpha, requires_grad=True)

        self.latent_mean = torch.nn.Parameter(data=torch.randn((90000, 3*self.N_domains), device=device), requires_grad=True)
        self.latent_std = torch.nn.Parameter(data=torch.ones((90000, 3*self.N_domains), device=device), requires_grad=True) #+ 0.1*torch.randn((90000, 3*self.N_domains)), requires_grad=True)
        #self.latent_std = torch.ones((90000, 3*self.N_domains))*0.001
        self.latent_mean = self.latent_mean.to(device)
        self.latent_std = self.latent_std.to(device)

    def multiply_windows_weights(self):
        weights_per_residues = torch.empty((self.N_residues, self.N_domains), device=self.device)
        for i in range(self.N_residues):
            windows_set = self.bs_per_res[i]  # Extracting the indexes of the windows for the given residue
            weights_per_residues[i, :] = torch.prod(self.weights[windows_set, :],
                                                    axis=0)  # Muliplying the weights of all the windows, for each subsystem

        #weights_per_residues = weights_per_residues**2
        #weights_per_residues = weights_per_residues/torch.sum(weights_per_residues, dim=0, keepdim=True)
        attention_softmax = F.softmax(weights_per_residues, dim=1)
        #div = torch.transpose(attention_softmax, 0, 1)/(torch.sum(attention_softmax, dim=1) + 1e-10)
        #attention_softmax = torch.transpose(div, 0, 1)
        #attention_softmax = F.softmax(weights_per_residues, dim=1)
        return attention_softmax


    def sample_q(self, indexes):
        latent_var = self.latent_std[indexes, :]*torch.randn((self.batch_size, 3*self.N_domains), device=self.device) + self.latent_mean[indexes, :]
        #latent_var = self.latent_std*torch.randn((self.batch_size, 3*self.N_domains)) + self.latent_mean
        return latent_var

    def func(self):
        attention_softmax = self.multiply_windows_weights()
        attention_softmax_log = torch.log(attention_softmax)
        prod = attention_softmax_log * attention_softmax
        loss = -torch.sum(prod)

        return loss / self.N_residues

    def deform_structure(self, weights, translation_scalars):
        """

        :param weights: weights of the attention mask tensor (N_residues, N_domains)
        :param translation_scalars: translations scalars used to compute translation vectors:
                tensor (Batch_size, N_domains, 3)
        :return: tensor (Batch_size, 3*N_residues, 3) corresponding to translated structure, tensor (3*N_residues, 3)
                of translation vectors

        Note that self.local_frame is a tensor of shape (3,3) with orthonormal vectors as rows.
        """
        ## Weighted sum of the local frame vectors, torch boracasts local_frame.
        ## Translation_vectors is (Batch_size, N_domains, 3)
        translation_vectors = torch.matmul(translation_scalars, self.local_frame)
        ## Weighted sum of the translation vectors using the mask. Outputs a translation vector per residue.
        ## translation_per_residue is (Batch_size, N_residues, 3)
        translation_per_residue = torch.matmul(weights, translation_vectors)
        ## We displace the structure, using an interleave because there are 3 consecutive atoms belonging to one
        ## residue.
        new_atom_positions = self.atom_absolute_positions + torch.repeat_interleave(translation_per_residue, 3, 1)
        return new_atom_positions, translation_per_residue

    def forward(self, x, edge_index, edge_attr, latent_variables):
        """

        :param x:
        :param edge_index:
        :param edge_attr:
        :param latent_variables: tensor of size (Batch_size, dim_latent)
        :return: tensors: a new structure (N_residues, 3), the attention mask (N_residues, N_domains),
                translation vectors for each residue (N_residues, 3) leading to the new structure.
        """
        batch_size = latent_variables.shape[0]
        weights = self.multiply_windows_weights()
        #hidden_representations = self.decoder.forward(x, edge_index, edge_attr, latent_variables)
        #weights_transp = torch.transpose(weights, 0, 1)
        #features_domain = torch.matmul(weights_transp, hidden_representations)
        #features_and_latent = torch.concat([features_domain, torch.broadcast_to(latent_variables,(1, 3))], dim=1)

        features_and_latent = latent_variables
        #features_and_latent = torch.broadcast_to(latent_variables, (1, 3))
        ## MLP computing the deformation scalars. Outputs 3 scalars per domain (Batch_size, 3*N_domains)
        scalars_per_domain = self.mlp_translation.forward(features_and_latent)
        scalars_per_domain = torch.reshape(scalars_per_domain, (batch_size, self.N_domains,3))
        new_structure, translations = self.deform_structure(weights, scalars_per_domain)
        return new_structure, weights, translations


    def loss(self, new_structure, true_deformation, mask_weights, indexes, train=True):
        """

        :param new_structure: tensor (3*N_residues, 3) of absolute positions of atoms.
        :param true_deformation: tensor (Batch_size, N_domains, 3) of true deformation vectors, one per domain.
        :param mask_weights: tensor (N_residues, N_domains), attention mask.
        :return: the RMSD loss and the entropy loss
        """
        ## true_deformed_structure is tensor (Batch_size, 3*N_residues, 3) containing absolute positions of every atom
        ## for each structure of the batch.
        #batch_size = true_deformation.shape[0]
        batch_size = self.batch_size
        true_deformed_structure = torch.empty((batch_size, 3*self.N_residues, 3), device=self.device)
        true_deformed_structure[:, :3*self.cutoff1, :] = self.atom_absolute_positions[:3*self.cutoff1, :] + true_deformation[:, 0:1, :]#**3
        true_deformed_structure[:, 3 * self.cutoff1:3*self.cutoff2, :] = self.atom_absolute_positions[3 * self.cutoff1:3*self.cutoff2, :] + true_deformation[:, 1:2, :]#**3
        true_deformed_structure[:, 3 * self.cutoff2:, :] = self.atom_absolute_positions[3 * self.cutoff2:, :] + true_deformation[:, 2:3, :]#**3
        rmsd = torch.mean(torch.sqrt(torch.mean(torch.sum((new_structure - true_deformed_structure)**2, dim=2), dim=1)))

        attention_softmax_log = torch.log(mask_weights+self.epsilon_mask_loss)
        prod = attention_softmax_log * mask_weights
        loss = -torch.sum(prod)

        Dkl_loss = - 0.5* torch.mean(torch.sum(1+torch.log(self.latent_std[indexes, :]**2) - self.latent_mean[indexes, :]**2
                                    - self.latent_std[indexes, :]**2, axis=1))

        #Dkl_loss = -0.5*torch.sum(1 + torch.log(self.latent_std[indexes, :]**2) - torch.log(torch.tensor(16**2)) - (self.latent_mean[indexes, :]**2
        #                            - self.latent_std[indexes, :]**2)/torch.tensor(16**2))

        #Dkl_loss = -0.5*torch.sum(1 + torch.log(self.latent_std**2) - torch.log(torch.tensor(16**2)) - (self.latent_mean**2
        #                            - self.latent_std**2)/torch.tensor(16**2))
        #loss_weights = F.softmax(mask_weights, dim=0)
        #loss = -torch.sum((loss_weights - 1/self.N_residues)**2)
        #loss = -torch.sum(torch.minimum(torch.sum(mask_weights, dim=0), torch.ones((self.N_domains))))
        #print("RMSD:", rmsd)
        if train:
            print("RMSD:", rmsd)
            print("Loss:", loss)
            print("Dkl:", Dkl_loss)
            #return 0.001*rmsd + loss #+ self.alpha_entropy*loss / self.N_residues
            #return rmsd + Dkl_loss + 0.000001*loss, rmsd, Dkl_loss, loss
            return rmsd + Dkl_loss, rmsd, Dkl_loss, loss
            #return Dkl_loss
            #return rmsd #+ 0.01*loss + 0.001*loss2

        return rmsd



