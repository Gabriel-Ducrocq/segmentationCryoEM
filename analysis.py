import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import utils
from imageRenderer import Renderer

dataset_path="data/vaeContinuousNoisyZhongStyleNoCTF/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
#This represent the number of true domains
N_domains = 3
N_pixels = 64*64
#This represents the number of domain we think there are
N_input_domains = 4
latent_dim = 9
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
model_path = "data/vaeContinuousNoisyZhongStyleNoCTF/full_model4630"
model = torch.load(model_path, map_location=torch.device(device))


#training_set = torch.load(dataset_path + "training_set.npy", map_location=torch.device(device)).to(device)
#training_rotations_angles = torch.load(dataset_path + "training_rotations_angles.npy", map_location=torch.device(device)).to(device)
#training_rotations_axis = torch.load(dataset_path + "training_rotations_axis.npy", map_location=torch.device(device)).to(device)
training_rotations_matrices = torch.load(dataset_path + "rotationPoseDataSet", map_location=torch.device(device)).to(device)
#training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices.npy", map_location=torch.device(device))
#noise_component = torch.load(dataset_path + "noise_component", map_location=torch.device(device))

training_images = torch.load("data/vaeContinuousNoisyZhongStyleNoCTF/continuousConformationDataSet")

#print("NOISE")
#print(torch.median(0.5*torch.sum(noise_component**2, dim=(-2,-1))))
#print(torch.mean(0.5*torch.sum(noise_component**2, dim=(-2,-1))))
#print(torch.max(0.5*torch.sum(noise_component**2, dim=(-2,-1))))
#print(torch.min(0.5*torch.sum(noise_component**2, dim=(-2,-1))))

print("SHOULD WE USE ENCODER:", model.use_encoder)
training_indexes = torch.tensor(np.array(range(10000)))
all_latent_distrib = []
all_indexes = []
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
        batch_rotation_matrices = training_rotations_matrices[batch_indexes]
        batch_deformed_images = training_images[batch_indexes]
        ## Deforming the structure for each batch data point

        print("Deformed")
        ## We then rotate the structure and project them on the x-y plane.
        latent_distrib = model.encode(batch_deformed_images)
        all_latent_distrib.append(latent_distrib.detach().numpy())
        all_indexes.append(batch_indexes.detach().numpy())


np.save(dataset_path + "latent_distrib.npy", np.array(all_latent_distrib))
np.save(dataset_path + "indexes.npy", np.array(all_indexes))