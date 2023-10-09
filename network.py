import torch
#import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np

import utils
from mlp import MLP
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix


class Net(torch.nn.Module):
    def __init__(self, N_residues, N_domains, latent_dim_x, encoder, decoder, renderer, local_frame, atom_absolute_positions,
                 batch_size, device, alpha_entropy = 0.0001, use_encoder = True, one_latent_per_domain=False,
                 encoder_x = None, nets_x_given_w=None, latent_clustering=False, latent_dim_w = 10,
                 n_components_mixture = 10):
        super(Net, self).__init__()
        self.encoder_x = encoder_x
        self.latent_dim_w = latent_dim_w
        self.nets_x_given_w = nets_x_given_w
        self.latent_clustering = latent_clustering
        self.n_components_mixture = n_components_mixture
        self.one_latent_per_domain = one_latent_per_domain
        self.N_residues = N_residues
        self.N_domains = N_domains
        self.latent_dim_x = latent_dim_x
        self.epsilon_loss = 1e-10
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
        self.device = device
        self.latent_mean = torch.nn.Parameter(data=torch.randn((100000, self.latent_dim_w), device=device), requires_grad=True)
        self.tau = 0.05
        self.annealing_tau = 1
        self.use_encoder = use_encoder
        #self.cluster_means = torch.nn.Parameter(data=torch.tensor([160, 550, 800, 1300], dtype=torch.float32,device=device)[None, :],
        #                                        requires_grad=True)
        #self.cluster_std = torch.nn.Parameter(data=torch.tensor([100, 100, 100, 100], dtype=torch.float32, device=device)[None, :],
        #                                      requires_grad=True)
        #self.cluster_proportions = torch.nn.Parameter(torch.ones(4, dtype=torch.float32, device=device)[None, :],
        #                                              requires_grad=True)
        self.residues = torch.arange(0, self.N_residues, 1, dtype=torch.float32, device=device)[:, None]

        self.cluster_means_mean = torch.nn.Parameter(data=torch.tensor([50, 150, 400, 700, 850, 950], dtype=torch.float32,device=device)[None, :],
                                                requires_grad=True)

        self.cluster_means_std = torch.nn.Parameter(data=torch.tensor([10, 10, 10, 10, 10, 10], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.cluster_std_mean = torch.nn.Parameter(data=torch.tensor([100, 100, 100, 100, 100, 100], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.cluster_std_std = torch.nn.Parameter(data=torch.tensor([10, 10, 10, 10, 10, 10], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.cluster_proportions_mean = torch.nn.Parameter(torch.zeros(self.N_domains, dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)

        self.cluster_proportions_std = torch.nn.Parameter(torch.ones(self.N_domains, dtype=torch.float32, device=device)[None, :],
                           requires_grad=True)


        self.cluster_parameters = {"means":{"mean":self.cluster_means_mean, "std":self.cluster_means_std},
                                   "stds":{"mean":self.cluster_std_mean, "std":self.cluster_std_std},
                                   "proportions":{"mean":self.cluster_proportions_mean, "std":self.cluster_proportions_std}}


        self.cluster_prior_means_mean = torch.tensor([50, 100, 300, 500, 700, 900], dtype=torch.float32,device=device)[None, :]
        self.cluster_prior_means_std = torch.tensor([100, 100, 100, 100, 100, 100], dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_std_mean = torch.tensor([100, 100, 100, 100, 100, 100], dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_std_std = torch.tensor([10, 10, 10, 10, 10, 10], dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_proportions_mean = torch.zeros(self.N_domains, dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_proportions_std = torch.ones(self.N_domains, dtype=torch.float32, device=device)[None, :]

        self.cluster_prior = {"means":{"mean":self.cluster_prior_means_mean, "std": self.cluster_prior_means_std},
                                   "stds":{"mean":self.cluster_prior_std_mean, "std": self.cluster_prior_std_std},
                                   "proportions":{"mean":self.cluster_prior_proportions_mean,"std":self.cluster_prior_proportions_std}}


    def compute_mask(self):
        cluster_proportions = torch.randn(self.N_domains, device=self.device)*self.cluster_proportions_std + self.cluster_proportions_mean
        cluster_means = torch.randn(self.N_domains, device=self.device)*self.cluster_means_std + self.cluster_means_mean
        cluster_std = torch.randn(self.N_domains, device=self.device)*self.cluster_std_std + self.cluster_std_mean
        proportions = torch.softmax(cluster_proportions, dim=1)
        log_num = -0.5*(self.residues - cluster_means)**2/cluster_std**2 + \
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

    def encode_x_given_y(self, images):
        """
        Encode images into latent variables
        :param images: (N_batch, N_pix_x, N_pix_y) containing the cryoEM images
        :return: (N_batch, 2*latent_dim_w) predicted gaussian distribution over the latent variables
        """
        flattened_images = torch.flatten(images, start_dim=1, end_dim=2)
        w_mean_std = self.encoder_x(flattened_images)
        latent_mean = w_mean_std[:, :self.latent_dim_x]
        latent_std = w_mean_std[:, self.latent_dim_x:]
        w = torch.randn(size=(self.batch_size, self.latent_dim_w), device=self.device)*latent_std + latent_mean
        return w, latent_mean, latent_std

    def compute_p_z_given_xw(self, x, w):
        """
        Compute the probabilities of the discrete variable z, z=k is x belongs to component K in the mixture
        :param x: torch.tensor(N_batch, latent_dim), latent variable
        :param w: torch.tensor(N_batch, latent_w), latent variable for generating x
        :return: torch.tensor(N_batch, N_components_mixture), probabilities that z = K
        """
        mean_std_per_component = torch.concat([net(w)[:, None, :] for net in self.nets_x_given_w], dim=1)
        #gaussian_pdf is a torch.tensor of size (N_batch, N_component_mixture)
        log_gaussian_pdfs = utils.compute_log_gaussian_pdf(x, mean_std_per_component, self.latent_dim_x)
        p_z = torch.softmax(log_gaussian_pdfs, dim=-1)
        return p_z

    def sample_latent(self, images=None):
        """
        Sample from the approximate posterior over the latent space
        :param distrib_parameters: (N_batch, 2*latent_dim) the parameters mu, sigma of the approximate posterior
        :return: (N_batch, latent_dim) actual samples.
        """
        batch_size = images.shape[0]
        latent_mean_std = self.encode(images)
        if self.one_latent_per_domain:
            latent_mean = latent_mean_std[:, :self.latent_dim_w*self.N_domains]
            latent_std = latent_mean_std[:, self.latent_dim_w*self.N_domains:]
            latent_vars = latent_std*torch.randn(size=(batch_size, self.latent_dim_w*self.N_domains), device=self.device) + latent_mean
        else:
            latent_mean = latent_mean_std[:, :self.latent_dim_w]
            latent_std = latent_mean_std[:, self.latent_dim_w:]
            print(latent_std.shape)
            print(latent_mean.shape)
            latent_vars = latent_std*torch.randn(size=(batch_size, self.latent_dim_w), device=self.device) + latent_mean

        return latent_vars, latent_mean, latent_std



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
        :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
        """
        #NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
        rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions)
        mask_rotation_per_domains_axis_angle = mask[None, :, :, None]*rotation_per_domains_axis_angle[:, None, :, :]

        mask_rotation_matrix_per_domain_per_residue = axis_angle_to_matrix(mask_rotation_per_domains_axis_angle)
        #Transposed here because pytorch3d has right matrix multiplication convention.
        #mask_rotation_matrix_per_domain_per_residue = torch.transpose(mask_rotation_matrix_per_domain_per_residue, dim0=-2, dim1=-1)
        overall_rotation_matrices = torch.zeros((self.batch_size, self.N_residues,3,3), device=self.device)
        overall_rotation_matrices[:, :, 0, 0] = 1
        overall_rotation_matrices[:, :, 1, 1] = 1
        overall_rotation_matrices[:, :, 2, 2] = 1
        for i in range(self.N_domains):
            overall_rotation_matrices = torch.matmul(mask_rotation_matrix_per_domain_per_residue[:, :, i, :, :],
                                                     overall_rotation_matrices)

        return overall_rotation_matrices

    def compute_Dkl_mask(self, variable):
        """
        Compute the Dkl loss between the prior and the approximated posterior distribution
        :param variable: string, either "proportions", "means" or "std"
        :return: Dkl loss
        """
        assert variable in ["means", "stds", "proportions"]
        return torch.sum(-1/2 + torch.log(self.cluster_prior[variable]["std"]/self.cluster_parameters[variable]["std"]) \
        + (1/2)*(self.cluster_parameters[variable]["std"]**2 +
        (self.cluster_prior[variable]["mean"] - self.cluster_parameters[variable]["mean"])**2)/self.cluster_prior[variable]["std"]**2)

    def compute_z_prior_term(self, x_mean, x_std, w_mean, w_std):
        """
        Compute the z prior term
        :param x_mean: torch.tensor(N_batch, latent_dim) of means of the approximate posterior over x
        :param x_std: torch.tensor(N_batch, latent_dim) of std of the approximate posterior over x
        :param w_mean: torch.tensor(N_batch, latent_dim_w) of mean of the approximate posterior over w
        :param w_std: torch.tensor(N_batch, latent_dim_w) of std of the approximate posterior over w
        :return: torch.tensor(N_batch, ) of the expected KL divergence estimated with ONE data point for each image.
        """
        x = torch.randn_like(x_mean, device=self.device)*x_std + x_mean
        w = torch.randn_like(w_mean, device=self.device)*w_std + w_mean
        #p_z is torch.tensor(N_batch, N_components_mixture)
        p_z = self.compute_p_z_given_xw(x,w)
        log_pz = torch.log(p_z)
        KL = torch.sum((log_pz + np.log(self.n_components_mixture))*p_z, dim=-1)
        return KL

    def compute_x_prior_term(self, x_mean, x_std, w_mean, w_std):
        """
        Compute the x prior term, approximated with only ONE point for each example in the batch
        :param x_mean: torch.tensor(N_batch, latent_dim)
        :param x_std: torch.tensor(N_batch, latent_dim)
        :param w_mean: torch.tensor(N_batch, latent_dim_w)
        :param w_std: torch.tensor(N_batch, latent_dim_w)
        :return: torch.tensor(N_batch, ) x prior term for each batch point
        """
        posterior_x_entropy = - self.latent_dim_x/2*np.log(2*np.pi) - torch.sum(1 + torch.log(x_std**2), dim=-1)*0.5

        x_sampled = x_mean + torch.randn_like(x_mean, device=self.device)*x_std
        w_sampled = w_mean + torch.randn_like(w_mean, device=self.device)*w_std
        #distributions_x_given_w is torch.tensor(N_batch, N_components_mixture, 2*latent_dim)
        distributions_x_given_w = torch.concat([net(w_sampled)[:, None, :] for net in self.nets_x_given_w], dim=1)
        log_p_x_given_wz = utils.compute_log_gaussian_pdf(x_sampled, distributions_x_given_w, self.latent_dim_x)
        p_z = self.compute_p_z_given_xw(x_sampled, w_sampled)
        cross_entropy_approx = torch.sum(log_p_x_given_wz*p_z, dim=-1)

        return posterior_x_entropy - cross_entropy_approx




    def forward(self, indexes, images=None):
        """
        Encode then decode image
        :images: (N_batch, N_pix_x, N_pix_y) cryoEM images
        :return: tensors: a new structure (N_batch, N_residues, 3), the attention mask (N_residues, N_domains),
                translation vectors for each residue (N_batch, N_residues, 3) leading to the new structure.
        """
        batch_size = indexes.shape[0]
        weights = self.compute_mask()
        latent_variables_w, latent_mean_w, latent_std_w = self.sample_latent(images)
        latent_variables_x, latent_mean_x, latent_std_x = self.encode_x_given_y(images)

        #features = torch.cat([latent_variables, rotation_angles, rotation_axis], dim=1)
        if not self.one_latent_per_domain:
            output = self.decoder.forward(latent_variables_x)
            ## The translations are the first 3 scalars and quaternions the last 3
            output = torch.reshape(output, (batch_size, self.N_domains, 2 * 3))
        else:
            output_all_domains = []
            for k in range(self.N_domains):
                latent_variable_k = latent_variables_x[:, self.latent_dim_x*k:self.latent_dim_x*(k+1)]
                output_k = self.decoder.forward(latent_variable_k)
                output_all_domains.append(output_k[:, None, :])

            output = torch.concat(output_all_domains, dim=1)

        scalars_per_domain = output[:, :, :3]
        ones = torch.ones(size=(self.batch_size, self.N_domains, 1), device=self.device)
        quaternions_per_domain = torch.cat([ones,output[:, :, 3:]], dim=-1)
        rotations_per_residue = self.compute_rotations(quaternions_per_domain, weights)
        new_structure, translations = self.deform_structure(weights, scalars_per_domain, rotations_per_residue)
        return new_structure, weights, translations, latent_variables_w, latent_mean_w, latent_std_w, latent_variables_x, latent_mean_x, latent_std_x


    def loss(self, new_structures, mask_weights, images, distrib_parameters, rotation_matrices, latent_mean_w=None, latent_std_w=None,
             x_mean=None, x_std=None, train=True):
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

        minus_batch_Dkl_loss = 0.5 * torch.sum(1 + torch.log(latent_std_w ** 2 + self.epsilon_loss) \
                                               - latent_mean_w ** 2 \
                                               - latent_std_w ** 2, dim=1)

        minus_batch_KL_x_prior = -self.compute_x_prior_term(x_mean, x_std, latent_mean_w, latent_std_w)
        minus_batch_KL_z_prior = -self.compute_z_prior_term(x_mean, x_std, latent_mean_w, latent_std_w)


        minus_batch_Dkl_mask_mean = -self.compute_Dkl_mask("means")
        minus_batch_Dkl_mask_std = -self.compute_Dkl_mask("stds")
        minus_batch_Dkl_mask_proportions = -self.compute_Dkl_mask("proportions")
        Dkl_loss = -torch.mean(minus_batch_Dkl_loss)

        #total_loss_per_batch = -batch_ll - 0.001*minus_batch_Dkl_loss
        #loss = torch.mean(total_loss_per_batch) - 0.0001*minus_batch_Dkl_mask_mean - 0.0001*minus_batch_Dkl_mask_std \
        #       - 0.0001*minus_batch_Dkl_mask_proportions

        #total_loss_per_batch = -batch_ll - 0.01*minus_batch_Dkl_loss
        #total_loss_per_batch = -batch_ll - 0.0001*minus_batch_Dkl_loss
        ##Trying with even lower weight on DKL:
        #total_loss_per_batch = -batch_ll - 0.0000001 * minus_batch_Dkl_loss
        total_loss_per_batch = -batch_ll - 0.001 * minus_batch_Dkl_loss - 0.001*minus_batch_KL_x_prior - 0.001*minus_batch_KL_z_prior

        l2_pen = 0
        for name,p in self.named_parameters():
            if "weight" in name and ("encoder" in name or "decoder" in name or "net" in name):
                l2_pen += torch.sum(p**2)

        for net in self.nets_x_given_w:
            for name, p in net.named_parameters():
                if "weight" in name:
                    l2_pen += torch.sum(p ** 2)

        loss = torch.mean(total_loss_per_batch) - 0.0001*minus_batch_Dkl_mask_mean - 0.0001*minus_batch_Dkl_mask_std \
               - 0.0001*minus_batch_Dkl_mask_proportions+0.001*l2_pen



        if train:
            print("Mask", mask_weights)
            print("RMSD:", nll)
            print("Dkl:", Dkl_loss)
            print("DKLS:", minus_batch_Dkl_mask_mean, minus_batch_Dkl_mask_proportions, minus_batch_Dkl_mask_std)
            return loss, nll, Dkl_loss, -minus_batch_Dkl_mask_mean, -minus_batch_Dkl_mask_std, -minus_batch_Dkl_mask_proportions

        return nll



