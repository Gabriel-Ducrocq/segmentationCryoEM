import numpy as np
import Bio.PDB as bpdb
import torch
import scipy
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3',
            'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O',
            'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}




aa_types= np.array((
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL'
))

def norm(u):
    """
    Computes the euclidean norm of a vector
    :param u: vector
    :return: euclidean norm of u
    """
    return np.sqrt(np.sum(u**2))

def gram_schmidt(u1, u2):
    """
    Orthonormalize a set of two vectors.
    :param u1: first non zero vector, unnormalized
    :param u2: second non zero vector, unormalized
    :return: orthonormal basis
    """
    e1 = u1/norm(u1)
    e2 = u2 - np.dot(u2, e1)*e1
    e2 /= norm(e2)
    return e1, e2

def get_orthonormal_basis(u1, u2):
    """
    Computes the local orthonormal frame basis based on the Nitrogen, C alpha and C beta atoms.
    :param u1: first non zero vector, unnormalized (here the bond vector between Nitrogen and C alpha carbon)
    :param u2: second non zero vector, unormalized  (here the bond vector between C beta and C alpha carbon)
    :return: An array of three orthonormal vectors of size 3, in colums
    """

    e1, e2 = gram_schmidt(u1, u2)
    e3 = np.cross(e1, e2)
    return np.array([e1, e2, e3]).T

def aa_one_hot(name):
    """
    Create the one-hot encoding vector for a specific amino-acid
    :param name: string, name of the amino acid in PDB format (e.g ALA for alanine)
    :return: np.array, type int, one hot encoding of the amino acid.
    """
    if name not in aa_types:
        raise ValueError("This amino-acid is unknown")

    return np.array(aa_types == name, dtype=int)


def compute_distance_matrix(locations):
    """
    Compute the distance matrix for all residue pairs.
    :param locations: numpy array of size (N_residues,3) of all C alpha positions
    :return: a symmetric numpy array of size (N_residues, N_residues) of pairwise distances
    """
    N_residues = len(locations)
    print(N_residues)
    distance_matrix = np.zeros((N_residues, N_residues))
    for i, pos in enumerate(locations):
        distance_matrix[i,i] = np.inf
        for j in range(i+1, N_residues):
            distance_matrix[i, j] = norm(pos - locations[j,:])

    distance_matrix += distance_matrix.T
    return distance_matrix


def find(ar, condition):
    """

    :param ar: array from which we want the indexes
    :param condition: condition upon which we get the index
    :return: the set of indexes verifying the condition
    """
    return [i for i,val in enumerate(ar) if condition(val)]


def gaussian_rbf(x,y,sigma):
    return np.exp(-(1/2)*np.sum((x-y)**2)/sigma**2)


def get_positions(residue, name):
    x = residue["CA"].get_coord()
    y = residue["N"].get_coord()
    if name == "GLY":
        z = residue["C"].get_coord()
        return x,y,z

    z = residue["C"].get_coord()
    return x,y,z


def compute_rotations(quaternions, mask, device):
    """
    Computes the rotation matrix corresponding to each domain for each residue, where the angle of rotation has been
    weighted by the mask value of the corresponding domain.
    :param quaternions: tensor (N_batch, N_domains, 4) of non normalized quaternions defining rotations
    :param mask: tensor (N_residues, N_input_domains)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """
    batch_size = quaternions.shape[0]
    N_residues = mask.shape[0]
    N_domains = mask.shape[1]
    #NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions)
    mask_rotation_per_domains_axis_angle = mask[None, :, :, None]*rotation_per_domains_axis_angle[:, None, :, :]

    mask_rotation_matrix_per_domain_per_residue = axis_angle_to_matrix(mask_rotation_per_domains_axis_angle)
    #Transposed here because pytorch3d has right matrix multiplication convention.
    #mask_rotation_matrix_per_domain_per_residue = torch.transpose(mask_rotation_matrix_per_domain_per_residue, dim0=-2, dim1=-1)
    overall_rotation_matrices = torch.zeros((batch_size, N_residues,3,3), device=device)
    overall_rotation_matrices[:, :, 0, 0] = 1
    overall_rotation_matrices[:, :, 1, 1] = 1
    overall_rotation_matrices[:, :, 2, 2] = 1
    for i in range(N_domains):
        overall_rotation_matrices = torch.matmul(mask_rotation_matrix_per_domain_per_residue[:, :, i, :, :],
                                                 overall_rotation_matrices)

    return overall_rotation_matrices

def deform_structure(atom_relative_positions, mask, translation_scalars, rotations_per_residue, local_frame,
                     local_frame_in_colums):
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
    batch_size = translation_scalars.shape[0]
    N_residues = mask.shape[0]
    ## Weighted sum of the local frame vectors, torch boracasts local_frame.
    ## Translation_vectors is (Batch_size, N_domains, 3)
    translation_vectors = torch.matmul(translation_scalars, local_frame)
    ## Weighted sum of the translation vectors using the mask. Outputs a translation vector per residue.
    ## translation_per_residue is (Batch_size, N_residues, 3)
    translation_per_residue = torch.matmul(mask, translation_vectors)
    ## We displace the structure, using an interleave because there are 3 consecutive atoms belonging to one
    ## residue.
    ##We compute the rotated frame for each residues, still set at the origin.
    rotated_frame_per_residue = torch.matmul(rotations_per_residue, local_frame_in_colums)
    rotated_frame_per_residue = torch.transpose(rotated_frame_per_residue, dim0=-2, dim1=-1)
    ##Given the rotated frames and the atom positions in these frames, we recover the transformed absolute positions
    ##### I think I should transpose the rotated frame before computing the new positions.
    transformed_absolute_positions = torch.matmul(torch.broadcast_to(atom_relative_positions,
                                                                     (batch_size, N_residues * 3, 3))[:, :,
                                                  None, :],
                                                  torch.repeat_interleave(rotated_frame_per_residue, 3, 1))
    new_atom_positions = transformed_absolute_positions[:, :, 0, :] + torch.repeat_interleave(translation_per_residue,
                                                                                              3, 1)
    return new_atom_positions, translation_per_residue




def process_structure(transform, atom_relative_positions, mask, local_frame, local_frame_in_colums, device, vae):
    N_domains = mask.shape[1]
    batch_size = transform.shape[0]
    transform = torch.reshape(transform, (batch_size, N_domains,2*3))
    scalars_per_domain = transform[:, :, :3]
    ones = torch.ones(size=(batch_size, N_domains, 1), device=device)
    quaternions_per_domain = torch.cat([ones,transform[:, :, 3:]], dim=-1)
    rotations_per_residue = compute_rotations(quaternions_per_domain, mask, device)
    #new_structure, translations = deform_structure(atom_relative_positions, mask, scalars_per_domain, rotations_per_residue,
    #                                               local_frame, local_frame_in_colums)
    new_structure, _ = vae.deform_structure(mask, scalars_per_domain, rotations_per_residue)
    return new_structure



def compute_Dkl_mask(network,variable):
    """
    Compute the Dkl loss between the prior and the approximated posterior distribution
    :param variable: string, either "proportions", "means" or "std"
    :return: Dkl loss
    """
    assert variable in ["means", "stds", "proportions"]
    return torch.sum(-1/2 + torch.log(network.cluster_prior[variable]["std"]/network.cluster_parameters[variable]["std"]) \
    + (1/2)*(network.cluster_parameters[variable]["std"]**2 +
    (network.cluster_prior[variable]["mean"] - network.cluster_parameters[variable]["mean"])**2)/network.cluster_prior[variable]["std"]**2)


def loss(vae, new_images, mask_weights, images, distrib_parameters, latent_mean=None, latent_std=None
         , train=True):
    """

    :param new_structures: tensor (N_batch, 3*N_residues, 3) of absolute positions of atoms.
    :images: tensor (N_batch, N_pix_x, N_pix_y) of cryoEM images
    :distrib_parameters: tensor (N_batch, 2*latent_dim) containing the mean and std of the distrib that
                        the encoder outputs.
    :return: the RMSD loss and the entropy loss
    """
    batch_ll = -0.5*torch.sum((new_images - images)**2, dim=(-2, -1))
    nll = -torch.mean(batch_ll)

    #if use_encoder:
    minus_batch_Dkl_loss = 0.5 * torch.sum(1 + torch.log(latent_std ** 2) \
                                           - latent_mean ** 2 \
                                           - latent_std ** 2, dim=1)

    #else:
    #    minus_batch_Dkl_loss = 0.5 * torch.sum(1 - torch.log(self.prior_std[distrib_parameters]**2) +
    #                                           torch.log(self.latent_std[distrib_parameters] ** 2)\
    #                                 - self.latent_mean[distrib_parameters] ** 2/ self.prior_std[distrib_parameters]**2 \
    #                                 - self.latent_std[distrib_parameters] ** 2/self.prior_std[distrib_parameters]**2, dim=1)


    minus_batch_Dkl_mask_mean = -compute_Dkl_mask(vae, "means")
    minus_batch_Dkl_mask_std = -compute_Dkl_mask(vae, "stds")
    minus_batch_Dkl_mask_proportions = -compute_Dkl_mask(vae, "proportions")
    Dkl_loss = -torch.mean(minus_batch_Dkl_loss)
    #total_loss_per_batch = -batch_ll - 0.001*minus_batch_Dkl_loss
    #loss = torch.mean(total_loss_per_batch) - 0.0001*minus_batch_Dkl_mask_mean - 0.0001*minus_batch_Dkl_mask_std \
    #       - 0.0001*minus_batch_Dkl_mask_proportions

    #total_loss_per_batch = -batch_ll - 0.01*minus_batch_Dkl_loss
    #total_loss_per_batch = -batch_ll - 0.0001*minus_batch_Dkl_loss
    ##Trying with even lower weight on DKL:
    #total_loss_per_batch = -batch_ll - 0.0000001 * minus_batch_Dkl_loss
    total_loss_per_batch = -batch_ll - 0.001 * minus_batch_Dkl_loss

    l2_pen = 0
    for name,p in vae.encoder.named_parameters():
        if "weight" in name and ("encoder" in name or "decoder" in name):
            l2_pen += torch.sum(p**2)

    loss = torch.mean(total_loss_per_batch) - 0.0001*minus_batch_Dkl_mask_mean - 0.0001*minus_batch_Dkl_mask_std \
           - 0.0001*minus_batch_Dkl_mask_proportions+0.001*l2_pen
    #else:
    #    loss = torch.mean(total_loss_per_batch) - 0.0001*minus_batch_Dkl_mask_mean - 0.0001*minus_batch_Dkl_mask_std \
    #           - 0.0001*minus_batch_Dkl_mask_proportions


    if train:
        print("Mask", mask_weights)
        print("RMSD:", nll)
        print("Dkl:", Dkl_loss)
        print("DKLS:", minus_batch_Dkl_mask_mean, minus_batch_Dkl_mask_proportions, minus_batch_Dkl_mask_std)
        return loss, nll, Dkl_loss, -minus_batch_Dkl_mask_mean, -minus_batch_Dkl_mask_std, -minus_batch_Dkl_mask_proportions

    return nll



