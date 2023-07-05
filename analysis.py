import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import utils
from pytorch3d.transforms import quaternion_to_axis_angle
from imageRenderer import Renderer

#dataset_path="data/vaeContinuousCTFNoisyBiModalAngle100kEncoder/"
dataset_path="../VAEProtein/data/vaeContinuousCTFNoisyBiModalAngleAngleTransition/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
#This represent the number of true domains
N_domains = 3
N_pixels = 240*240
#This represents the number of domain we think there are
N_input_domains = 4
latent_dim = 1
num_nodes = 1510
cutoff1 = 300
cutoff2 = 1353
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
#B = 10
B = 100
S = 1
dataset_size = 10000
test_set_size = int(dataset_size/10)

graph_file="data/features.npy"
features = np.load(graph_file, allow_pickle=True)
features = features.item()
absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
absolute_positions = absolute_positions.to(device)
local_frame = torch.tensor(features["local_frame"])
local_frame = local_frame.to(device)
relative_positions = torch.matmul(absolute_positions, local_frame)
pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device)
#model_path = "data/vaeContinuousCTFNoisyBiModalAngle100kEncoder/full_model"
model_path = "../VAEProtein/data/vaeContinuousCTFNoisyBiModalAngleAngleTransition/full_model"
model = torch.load(model_path, map_location=torch.device(device))


training_set = torch.load(dataset_path + "training_set", map_location=torch.device(device)).to(device)
training_rotations_angles = torch.load(dataset_path + "training_rotations_angles", map_location=torch.device(device)).to(device)
training_rotations_axis = torch.load(dataset_path + "training_rotations_axis", map_location=torch.device(device)).to(device)
training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices", map_location=torch.device(device)).to(device)
training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices", map_location=torch.device(device))
images = torch.load(dataset_path + "continuousConformationDataSet")

#print("SHOULD WE USE ENCODER:", model.use_encoder)
print("DATASET SIZE:", training_set.shape)
training_indexes = torch.tensor(np.array(range(10000)))
all_latent_distrib = []
all_indexes = []
all_axis_rotation = []
all_angle_rotation = []
all_translation = []

for epoch in range(0, 1):
    epoch_loss = torch.empty(1)
    # data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    data_loader = iter(DataLoader(training_indexes, batch_size=batch_size, shuffle=False))
    for i in range(100):
        start = time.time()
        print("epoch:", epoch)
        print(i / 100)
        # batch_data = next(iter(data_loader))
        batch_indexes = next(data_loader)
        print(batch_indexes)
        ##Getting the batch translations, rotations and corresponding rotation matrices
        batch_images = images[batch_indexes].to(device)
        batch_images = torch.flatten(batch_images, start_dim=1, end_dim=2)

        latent_distrib = model.encoder.forward(batch_images)
        latent_mean = latent_distrib[:, :latent_dim]
        output = model.decoder.forward(latent_mean)
        output = torch.reshape(output, (batch_size, N_input_domains, 2 * 3))
        scalars_per_domain = output[:, :, :3]
        ones = torch.ones(size=(batch_size, N_input_domains, 1), device=device)
        quaternions_per_domain = torch.cat([ones, output[:, :, 3:]], dim=-1)
        rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions_per_domain)
        angle_rotation = torch.sqrt(torch.sum(rotation_per_domains_axis_angle**2, dim = -1))
        axis_rotation = rotation_per_domains_axis_angle/angle_rotation[:, :, None]

        translation_vectors = torch.matmul(scalars_per_domain, local_frame)
        all_translation.append(translation_vectors.detach().cpu().numpy())

        all_latent_distrib.append(latent_distrib.detach().cpu().numpy())
        all_indexes.append(batch_indexes.detach().cpu().numpy())
        all_angle_rotation.append(angle_rotation.detach().cpu().numpy())
        all_axis_rotation.append(axis_rotation.detach().cpu().numpy())


np.save(dataset_path + "latent_distrib.npy", np.array(all_latent_distrib))
np.save(dataset_path + "indexes.npy", np.array(all_indexes))
np.save(dataset_path + "all_angle_rotation.npy", np.array(all_angle_rotation))
np.save(dataset_path + "all_axis_rotation.npy", np.array(all_axis_rotation))
np.save(dataset_path + "all_translations.npy", np.array(all_translation))