import torch
#import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np
from mlp import MLP
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix


class Net(torch.nn.Module):
    def __init__(self, N_residues, N_domains, latent_dim, B, S, encoder, decoder, renderer, local_frame, atom_absolute_positions,
                 batch_size, cutoff1, cutoff2, device, alpha_entropy = 0.0001):
        super(Net, self).__init__()
        self.N_residues = N_residues
        self.N_domains = N_domains
        self.B = B
        self.S = S
        self.latent_dim = latent_dim
        self.epsilon_mask_loss = 1e-10
        self.encoder = encoder
        self.decoder = decoder
        self.renderer = renderer
        self.local_frame_in_colums = local_frame
        ##Local frame is set with vectors in rows in the next line:
        self.local_frame = torch.transpose(local_frame, 0, 1)
        self.atom_absolute_positions = atom_absolute_positions
        ##Next line compute the coordinates in the local frame (N_atoms,3)
        self.relative_positions = torch.matmul(self.atom_absolute_positions, local_frame)
        self.batch_size = batch_size
        self.alpha_entropy = alpha_entropy
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.device = device
        self.SLICE_MU = slice(0,self.latent_dim)
        self.SLICE_SIGMA = slice(self.latent_dim, 2*self.latent_dim)

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
        ##No grad at first on the mask weights !!
        self.weights = torch.nn.Parameter(data=alpha, requires_grad=True)

        self.latent_mean = torch.nn.Parameter(data=torch.randn((10000, self.latent_dim)), requires_grad=True)
        self.latent_std = torch.nn.Parameter(data=torch.randn((10000, self.latent_dim)), requires_grad=True)
        #self.latent_std = torch.ones((90000, 3*self.N_domains))*0.001

        self.tau = 0.05
        
        #self.annealing_tau = 0.5
        self.annealing_tau = 1

        self.cluster_means = torch.nn.Parameter(data=torch.tensor([160, 550, 800, 1300], dtype=torch.float32,device=device)[None, :],
                                                requires_grad=True)
        self.cluster_std = torch.nn.Parameter(data=torch.tensor([100, 100, 100, 100], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)
        self.cluster_proportions = torch.nn.Parameter(torch.ones(4, dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)
        self.residues = torch.arange(0, 1510, 1, dtype=torch.float32, device=device)[:, None]


    def multiply_windows_weights(self):
        weights_per_residues = torch.empty((self.N_residues, self.N_domains), device=self.device)
        for i in range(self.N_residues):
            windows_set = self.bs_per_res[i]  # Extracting the indexes of the windows for the given residue
            weights_per_residues[i, :] = torch.prod(self.weights[windows_set, :],
                                                    dim=0)  # Muliplying the weights of all the windows, for each subsystem

        #weights_per_residues = weights_per_residues**2
        #weights_per_residues = weights_per_residues/torch.sum(weights_per_residues, dim=0, keepdim=True)
        attention_softmax = F.softmax(weights_per_residues/self.tau, dim=1)
        #div = torch.transpose(attention_softmax, 0, 1)/(torch.sum(attention_softmax, dim=1) + 1e-10)
        #attention_softmax = torch.transpose(div, 0, 1)
        #attention_softmax = F.softmax(weights_per_residues, dim=1)

        #attention_softmax = torch.zeros_like(attention_softmax)
        #attention_softmax[:300, 0] = 1
        #attention_softmax[300:1000, 1] = 1
        #attention_softmax[1000:, 2] = 1
        return attention_softmax

    def compute_mask(self):
        proportions = torch.softmax(self.cluster_proportions, dim=1)
        log_num = -0.5*(self.residues - self.cluster_means)**2/self.cluster_std**2 + \
              torch.log(proportions)

        mask = torch.softmax(log_num/self.tau, dim=1)
        return mask

    def encode(self, images):
        """
        Encode images into latent varaibles
        :param images: (N_batch, N_pix_x, N_pix_y) containing the cryoEM images
        :return: (N_batch, 2*N_domains) predicted gaussian distribution over the latent variables
        """
        flattened_images = torch.flatten(images, start_dim=1, end_dim=2)
        distrib_parameters = self.encoder.forward(flattened_images)
        return distrib_parameters
    def sample_latent(self, distrib_parameters):
        """
        Sample from the approximate posterior over the latent space
        :param distrib_parameters: (N_batch, 2*latent_dim) the parameters mu, sigma of the approximate posterior
        :return: (N_batch, latent_dim) actual samples.
        """
        batch_size = distrib_parameters.shape[0]
        #latent_vars = torch.randn(size=(batch_size, self.latent_dim), device=self.device)*distrib_parameters[:, self.SLICE_SIGMA]\
        #              + distrib_parameters[:, self.SLICE_MU]

        latent_vars = self.latent_std[distrib_parameters]*torch.randn(size=(batch_size, self.latent_dim), device=self.device)\
                      + self.latent_mean[distrib_parameters]
        return latent_vars


    def deform_structure(self, weights, translation_scalars, rotations_per_residue):
        """
        Note that the reference frame absolutely needs to be the SAME for all the residues (placed in the same spot),
         otherwise the rotation will NOT be approximately rigid !!!
        :param weights: weights of the attention mask tensor (N_residues, N_domains)
        :param translation_scalars: translations scalars used to compute translation vectors:
                tensor (Batch_size, N_domains, 3)
        :param rotations_per_residue: tensor (N_batch, N_res, 3, 3) of rotation matrices per residue
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
        ##We compute the rotated frame for each residues, still set at the origin.
        rotated_frame_per_residue = torch.matmul(rotations_per_residue, self.local_frame_in_colums)
        rotated_frame_per_residue = torch.transpose(rotated_frame_per_residue, dim0=-2, dim1=-1)
        ##Given the rotated frames and the atom positions in these frames, we recover the transformed absolute positions
        ##### I think I should transpose the rotated frame before computing the new positions.
        transformed_absolute_positions = torch.matmul(torch.broadcast_to(self.relative_positions,
                                                (self.batch_size, self.N_residues*3, 3))[:, :, None, :],
                                                      torch.repeat_interleave(rotated_frame_per_residue, 3, 1))
        #atom_abs_pos = torch.matmul(torch.broadcast_to(self.relative_positions,
        #                                                    (self.batch_size, self.N_residues*3, 3))[:, :, None, :],
        #                                              self.local_frame)
        new_atom_positions = transformed_absolute_positions[:, :, 0, :] + torch.repeat_interleave(translation_per_residue, 3, 1)
        #new_atom_positions = atom_abs_pos[:, :, 0, :] + torch.repeat_interleave(translation_per_residue, 3, 1)

        return new_atom_positions, translation_per_residue

    def compute_rotations(self, quaternions, mask):
        """
        Computes the rotation matrix corresponding to each domain for each residue, where the angle of rotation has been
        weighted by the mask value of the corresponding domain.
        :param quaternions: tensor (N_batch, N_domains, 4) of non normalized quaternions defining rotations
        :param mask: tensor (N_residues, N_input_domains)
        :return: tensor (N_batch, N_residues, N_input_domain, 3, 3) rotation matrix for each residue
        """
        #NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
        rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions)
        mask_rotation_per_domains_axis_angle = mask[None, :, :, None]*rotation_per_domains_axis_angle[:, None, :, :]

        mask_rotation_matrix_per_domain_per_residue = axis_angle_to_matrix(mask_rotation_per_domains_axis_angle)
        overall_rotation_matrices = torch.zeros((self.batch_size, self.N_residues,3,3))
        overall_rotation_matrices[:, :, 0, 0] = 1
        overall_rotation_matrices[:, :, 1, 1] = 1
        overall_rotation_matrices[:, :, 2, 2] = 1
        for i in range(self.N_domains):
            overall_rotation_matrices = torch.matmul(mask_rotation_matrix_per_domain_per_residue[:, :, i, :, :],
                                                     overall_rotation_matrices)

        return overall_rotation_matrices

    def forward(self, indexes, rotation_angles, rotation_axis):
        """
        Encode then decode image
        :images: (N_batch, N_pix_x, N_pix_y) cryoEM images
        :return: tensors: a new structure (N_batch, N_residues, 3), the attention mask (N_residues, N_domains),
                translation vectors for each residue (N_batch, N_residues, 3) leading to the new structure.
        """
        #batch_size = images.shape[0]
        batch_size = indexes.shape[0]
        #weights = self.multiply_windows_weights()
        weights = self.compute_mask()
        #distrib_parameters = self.encode(images)
        #latent_variables = self.sample_latent(distrib_parameters)
        latent_variables = self.sample_latent(indexes)
        features = torch.cat([latent_variables, rotation_angles, rotation_axis], dim=1)
        output = self.decoder.forward(features)
        ## The translations are the first 3 scalars and quaternions the last 3
        output = torch.reshape(output, (batch_size, self.N_domains,2*3))
        scalars_per_domain = output[:, :, :3]
        quaternions_per_domain = torch.cat([torch.ones(size=(self.batch_size, self.N_domains, 1)),output[:, :, 3:]],
                                           dim=-1)
        rotations_per_residue = self.compute_rotations(quaternions_per_domain, weights)

        #rotations_per_residue = torch.zeros((self.batch_size, self.N_residues, 3, 3))
        #rotations_per_residue[:, :, 0, 0] = 1
        #rotations_per_residue[:, :, 1, 1] = 1
        #rotations_per_residue[:, :, 2, 2] = 1
        new_structure, translations = self.deform_structure(weights, scalars_per_domain, rotations_per_residue)
        return new_structure, weights, translations, latent_variables


    def loss(self, new_structures, mask_weights, images, distrib_parameters, rotation_matrices, train=True):
        """

        :param new_structures: tensor (N_batch, 3*N_residues, 3) of absolute positions of atoms.
        :images: tensor (N_batch, N_pix_x, N_pix_y) of cryoEM images
        :distrib_parameters: tensor (N_batch, 2*latent_dim) containing the mean and std of the distrib that
                            the encoder outputs.
        :return: the RMSD loss and the entropy loss
        """
        new_images = self.renderer.compute_x_y_values_all_atoms(new_structures, rotation_matrices)
        batch_ll = -0.5*torch.sum((new_images - images)**2, dim=(-2, -1))
        nll = -torch.mean(batch_ll)


        attention_softmax_log = torch.log(mask_weights+self.epsilon_mask_loss)
        prod = attention_softmax_log * mask_weights
        loss_mask = -torch.sum(prod)

        #means = distrib_parameters[:, self.SLICE_MU]
        #std = distrib_parameters[:, self.SLICE_SIGMA]
        #batch_Dkl_loss = 0.5*torch.sum(1 + torch.log(std**2) - means**2 - std**2, dim=1)
        batch_Dkl_loss = 0.5 * torch.sum(1 + torch.log(self.latent_std[distrib_parameters] ** 2)\
                                         - self.latent_mean[distrib_parameters] ** 2 \
                                         - self.latent_std[distrib_parameters] ** 2, dim=1)
        Dkl_loss = -torch.mean(batch_Dkl_loss)
        total_loss_per_batch = -batch_ll - 0.001*batch_Dkl_loss
        loss = torch.mean(total_loss_per_batch)
        if train:
            print("RMSD:", nll)
            print("Dkl:", Dkl_loss)
            print("loss mask:", loss_mask)
            return loss, nll, Dkl_loss, loss_mask

        return nll



