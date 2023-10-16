import torch
import numpy as np
from scipy.optimize import dual_annealing
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix, axis_angle_to_quaternion

device = "cpu"
graph_file="../VAEProtein/data/vaeContinuousMD_open/features_open.npy"
features = np.load(graph_file, allow_pickle=True)
features = features.item()
absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
absolute_positions = absolute_positions.to(device)
local_frame = torch.tensor(features["local_frame"])
local_frame = local_frame.to(device)

all_images = torch.load("../VAEProtein/data/vaeContinuousMD_open/continuousConformationDataSet")
image = all_images[0]

def optim_function(transformation, image, mask, model, N_domains=6):
    rotation_axis_per_domain = torch.zeros((6, 3), dtype=torch.float32)
    transfomation_domains = torch.tensor(np.reshape(transformation, (N_domains, 6)), dtype=torch.float32)
    rotation_axis_per_domain[:, 0] = torch.sin(transfomation_domains[:, 0])*torch.cos(transfomation_domains[:, 1])
    rotation_axis_per_domain[:, 1] = torch.sin(transfomation_domains[:, 0])*torch.sin(transfomation_domains[:, 1])
    rotation_axis_per_domain[:, 2] = torch.cos(transfomation_domains[:, 0])

    rotation_axis_angle_per_domain = rotation_axis_per_domain*rotation_axis_per_domain[:, 2]
    scalars_per_domain = transfomation_domains[:, 3:]
    quaternions_per_domain = axis_angle_to_quaternion(rotation_axis_angle_per_domain)
    rotations_per_residue = model.compute_rotations(quaternions_per_domain[None, :, :], mask[:, :])
    new_structure, translations = model.deform_structure(mask, scalars_per_domain, rotations_per_residue)









