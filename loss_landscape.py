import torch
from network import Net
import numpy as np
from imageRenderer import Renderer
from pytorch3d.transforms import axis_angle_to_quaternion, matrix_to_axis_angle
import matplotlib.pyplot as plt

batch_size = 100
#This represent the number of true domains
N_domains = 3
N_pixels = 240*240
#This represents the number of domain we think there are
N_input_domains = 4
latent_dim = 1
num_nodes = 1510
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
dataset_size = 10000
test_set_size = int(dataset_size/10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "data/vaeContinuousCTFNoisyBiModalAngle/"
pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device, use_ctf=True)
training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices").to(device)
training_images = torch.load(dataset_path + "continuousConformationDataSet")
training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices")
ax_ang = matrix_to_axis_angle(training_conformation_rotation_matrix)
print("Axis angle SHAPE", ax_ang.shape)
print(ax_ang[:, 2, :])
angl = torch.sqrt(torch.sum(ax_ang[:, 2, :]**2, dim=-1))

graph_file="data/features.npy"

features = np.load(graph_file, allow_pickle=True)
features = features.item()
absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
absolute_positions = absolute_positions.to(device)
local_frame = torch.tensor(features["local_frame"])
local_frame = local_frame.to(device)
net = Net(num_nodes, N_input_domains, latent_dim, None, None, renderer, local_frame,
          absolute_positions, batch_size, device, use_encoder=False)
net.to(device)


mask = torch.zeros(size = (num_nodes, N_input_domains), dtype=torch.float32)
mask[1353:, -1] = 1.0
mask[:1353, 0] = 1.0

axis_rotation = torch.tensor(np.array([[[0, 1, 0], [0, 1, 0], [0, 1, 0], [0,  1,  0]]]), dtype=torch.float32)
angle_rotation = torch.zeros(size=(1000, 4), dtype=torch.float32)
##We rotate only the 4th domain to see
angle_rotation[:, 0] = 1.0
ang = torch.tensor(np.linspace(-np.pi, np.pi, 1000), dtype=torch.float32)[:, None]
angle_rotation *= ang
print(angle_rotation)

print(axis_rotation.shape)
print(angle_rotation.shape)

axis_angle = axis_rotation*angle_rotation[:, :, None]
quaternion = axis_angle_to_quaternion(axis_angle)
net.batch_size = 1000
rotation_per_residue = net.compute_rotations(quaternion, mask)
deformed_structures, _ = net.deform_structure(mask, torch.zeros(size=(1000, 4, 3)), rotation_per_residue)

image = training_images[9000]
rotation_matrices = training_rotations_matrices[9000]

plt.imshow(training_images[0], cmap="gray")
plt.show()
print(image.shape)
print(rotation_matrices.shape)
print(deformed_structures.shape)
new_images = renderer.compute_x_y_values_all_atoms(deformed_structures, rotation_matrices)
print("New images shape", new_images.shape)
all_losses = []
for i in range(1000):
    print(i)
    all_losses.append(torch.sum((new_images[i] - image)**2).detach().numpy())

all_losses = np.array(all_losses)
np.save("all_losses.npy", all_losses)

training_rotations_angles = torch.load(dataset_path + "training_rotations_angles").to(device)
print(angl[9000])

plt.plot(ang.detach().numpy(), all_losses)
plt.axvline(x=-2*np.pi/3)
plt.axvline(x=-np.pi/3)
plt.show()





