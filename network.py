import torch
#import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np
from mlp import MLP
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix


class Net(torch.nn.Module):
    def __init__(self, N_residues, N_domains, latent_dim, B, S, encoder, decoder, renderer, local_frame, atom_absolute_positions,
                 batch_size, cutoff1, cutoff2, device, alpha_entropy = 0.0001):
        super(Net, self).__init__()
        self.N_residues = N_residues
        self.N_domains = N_domains
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
        self.tau = 0.05
        self.annealing_tau = 1
        self.cluster_means = torch.nn.Parameter(data=torch.tensor([160, 550, 800, 1300], dtype=torch.float32,device=device)[None, :],
                                                requires_grad=True)
        self.cluster_std = torch.nn.Parameter(data=torch.tensor([100, 100, 100, 100], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)
        self.cluster_proportions = torch.nn.Parameter(torch.ones(4, dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)
        self.residues = torch.arange(0, 1510, 1, dtype=torch.float32, device=device)[:, None]

        self.dim = 3
        ##Forces:
        self.mean_translations = torch.zeros((N_domains, 3), dtype=torch.float32, device=device)
        self.std_translations = torch.ones((N_domains, 3), dtype=torch.float32, device=device)

        self.mean_quaternions = torch.zeros((N_domains, 3), dtype=torch.float32, device=device)
        self.std_quaternions = torch.ones((N_domains, 3), dtype=torch.float32, device=device)



    def compute_mask(self):
        proportions = torch.softmax(self.cluster_proportions, dim=1)
        log_num = -0.5*(self.residues - self.cluster_means)**2/self.cluster_std**2 + \
              torch.log(proportions)

        mask = torch.softmax(log_num/self.tau, dim=1)
        return mask

    def sample_forces(self, batch_indexes):
        """
        Sample from the approximate posterior over the latent space
        :param distrib_parameters: (N_batch, 2*latent_dim) the parameters mu, sigma of the approximate posterior
        :return: (N_batch, latent_dim) actual samples.
        """
        batch_size = batch_indexes.shape[0]
        samples_translations_scalars = self.mean_translations + torch.randn(size=(batch_size, self.N_domains, self.dim), device = self.device) \
                *self.std_translations

        samples_quaternions = self.mean_quaternions + torch.randn(size=(batch_size, self.N_domains, self.dim), device = self.device) \
                *self.std_quaternions

        return samples_translations_scalars, samples_quaternions


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
        new_atom_positions = transformed_absolute_positions[:, :, 0, :] + torch.repeat_interleave(translation_per_residue, 3, 1)
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
        #Transposed here because pytorch3d has right matrix multiplication convention.
        mask_rotation_matrix_per_domain_per_residue = torch.transpose(mask_rotation_matrix_per_domain_per_residue, dim0=-2, dim1=-1)
        overall_rotation_matrices = torch.zeros((self.batch_size, self.N_residues,3,3), device=self.device)
        overall_rotation_matrices[:, :, 0, 0] = 1
        overall_rotation_matrices[:, :, 1, 1] = 1
        overall_rotation_matrices[:, :, 2, 2] = 1
        for i in range(self.N_domains):
            overall_rotation_matrices = torch.matmul(mask_rotation_matrix_per_domain_per_residue[:, :, i, :, :],
                                                     overall_rotation_matrices)

        return overall_rotation_matrices

    def forward(self, indexes):
        """
        Encode then decode image
        :images: (N_batch, N_pix_x, N_pix_y) cryoEM images
        :return: tensors: a new structure (N_batch, N_residues, 3), the attention mask (N_residues, N_domains),
                translation vectors for each residue (N_batch, N_residues, 3) leading to the new structure.
        """
        weights = self.compute_mask()
        scalars_per_domain, quaternions = self.sample_forces(indexes)
        ones = torch.ones(size=(self.batch_size, self.N_domains, 1), device=self.device)
        quaternions_per_domain = torch.cat([ones,quaternions], dim=-1)
        rotations_per_residue = self.compute_rotations(quaternions_per_domain, weights)
        new_structure, translations = self.deform_structure(weights, scalars_per_domain, rotations_per_residue)
        return new_structure, weights, translations


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

        #We sum on the different latent force of the different domains and at the same time on the different
        batch_Dkl_loss_translations = 0.5 * torch.sum(1 + torch.log(self.std_translations ** 2)\
                                         - self.mean_translations ** 2 \
                                         - self.std_translations ** 2)

        batch_Dkl_loss_quaternions = 0.5 * torch.sum(1 + torch.log(self.std_quaternions ** 2)\
                                         - self.mean_quaternions ** 2 \
                                         - self.std_quaternions ** 2)


        Dkl_loss = -batch_Dkl_loss_translations - batch_Dkl_loss_quaternions
        total_loss_per_batch = -batch_ll - 0.001*batch_Dkl_loss_translations - 0.001*batch_Dkl_loss_quaternions
        loss = torch.mean(total_loss_per_batch)
        if train:
            print("RMSD:", nll)
            print("Dkl:", Dkl_loss)
            print("loss mask:", loss_mask)
            return loss, nll, Dkl_loss, loss_mask

        return nll



