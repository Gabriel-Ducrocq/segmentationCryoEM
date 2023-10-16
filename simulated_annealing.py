import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.optimize import dual_annealing
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix, axis_angle_to_quaternion
from imageRenderer import Renderer

N_domains = 6
device = "cpu"
graph_file="../VAEProtein/data/vaeContinuousMD_open/features_open.npy"
features = np.load(graph_file, allow_pickle=True)
features = features.item()
absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
absolute_positions = absolute_positions.to(device)
local_frame = torch.tensor(features["local_frame"])
local_frame = local_frame.to(device)


N_images = 100000
pixels_x = np.linspace(-70, 70, num=140).reshape(1, -1)
pixels_y = np.linspace(-70, 70, num=140).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device, use_ctf=True, N_heavy=3018)
model = torch.load("../VAEProtein/data/vaeContinuousMD_open/full_model2151", map_location=torch.device('cpu'))
model.device="cpu"
all_images = torch.load("../VAEProtein/data/vaeContinuousMD_open/continuousConformationDataSet", map_location=torch.device('cpu'))
all_poses = torch.load("../VAEProtein/data/vaeContinuousMD_open/training_rotations_matrices", map_location=torch.device('cpu'))
image = all_images[0]
pose = all_poses[0]
model.batch_size = 1

def optim_function(transformation, image, mask, model, N_domains=6):
    print("Iteration")
    rotation_axis_per_domain = torch.zeros((6, 3), dtype=torch.float32)
    transfomation_domains = torch.tensor(np.reshape(transformation, (N_domains, 6)), dtype=torch.float32)
    rotation_axis_per_domain[:, 0] = torch.sin(transfomation_domains[:, 0])*torch.cos(transfomation_domains[:, 1])
    rotation_axis_per_domain[:, 1] = torch.sin(transfomation_domains[:, 0])*torch.sin(transfomation_domains[:, 1])
    rotation_axis_per_domain[:, 2] = torch.cos(transfomation_domains[:, 0])
    rotation_angle_per_domain = transfomation_domains[:, 2]

    rotation_axis_angle_per_domain = rotation_axis_per_domain*rotation_angle_per_domain[:, None]
    scalars_per_domain = transfomation_domains[:, 3:]
    quaternions_per_domain = axis_angle_to_quaternion(rotation_axis_angle_per_domain)
    rotations_per_residue = model.compute_rotations(quaternions_per_domain[None, :, :], mask[:, :])
    new_structure, translations = model.deform_structure(mask, scalars_per_domain[None, :, :], rotations_per_residue)
    predicted_image = renderer.compute_x_y_values_all_atoms(new_structure, pose[None, :, :])
    loss = 0.5*torch.sum((predicted_image - image)**2)
    return loss.numpy()


transf = np.zeros(36,)
mask = torch.zeros((1006, 6), dtype=torch.float32)
mask[:124, 0] = 1
mask[124:320, 1] = 1
mask[320:506, 2] = 1
mask[506-824, 3] = 1
mask[824-865, 4] = 1
mask[865:1007, 5] = 1

bound_up = np.array([np.pi, 2*np.pi, np.pi, 10, 10, 10])
bound_low = np.array([0, 0, 0, -10, -10, -10])
all_bound_up = np.concatenate([bound_up for _ in range(N_domains)])
all_bound_low = np.concatenate([bound_low for _ in range(N_domains)])
all_bound = list(zip(all_bound_low, all_bound_up))
print("running dual annealing:")
loss = optim_function(np.zeros(36,), image, mask, model, 6)
print("Loss", loss)
res = dual_annealing(optim_function, all_bound, args=(image, mask, model, 6), no_local_search=True, initial_temp=10000, maxiter=100000)
print(np.reshape(res.x, (6,6)))
print(res.fun)
print(res.message)
#loss = optim_function(transf, image, mask, model)
#print(loss)










