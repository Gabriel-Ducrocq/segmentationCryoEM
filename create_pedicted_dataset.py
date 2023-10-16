import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from imageRenderer import Renderer
from pytorch3d.transforms import axis_angle_to_matrix
import utils
from imageRenderer import Renderer
from os import getcwd
from Bio.PDB.PDBParser import PDBParser


noise_var = 0.7
dataset_path = "../VAEProtein/data/vaeContinuousMD6DomainsDecoupledLatent/predicted_structures/"
#dataset_path = "data/test/"
print(getcwd())
N_input_domains = 6
batch_size=10
#batch_size=2
cutoff1 = 300
cutoff2 = 1353
graph_file = "../VAEProtein/data/vaeContinuousMD_open/features_open.npy"
device = "cpu"
features = np.load(graph_file, allow_pickle=True)
features = features.item()
absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
absolute_positions = absolute_positions.to(device)
local_frame = torch.tensor(features["local_frame"])
local_frame = local_frame.to(device)

N_images = 100000
pixels_x = np.linspace(-70, 70, num=140).reshape(1, -1)
pixels_y = np.linspace(-70, 70, num=140).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device, use_ctf=False, N_heavy=3018)

training_indexes = torch.tensor(np.array(range(N_images)))
all_images_predicted = []
parser = PDBParser(PERMISSIVE=0)
batch_rotation_matrices = torch.eye(3)[None, :, :].repeat(10, 1, 1)
path_predicted = dataset_path
all_rotation_matrices = torch.load("../VAEProtein/data/vaeContinuousMD6DomainsDecoupledLatent/training_rotations_matrices")
model = torch.load("../VAEProtein/data/vaeContinuousMD6DomainsDecoupledLatent/full_model2124", map_location=torch.device('cpu'))
model.device = "cpu"
model.batch_size = 100
"""
imageDataset = torch.load("../VAEProtein/data/vaeContinuousMD6DomainsDecoupledLatent/continuousConformationDataSet")
for epoch in range(0,1):
    for i in range(0, 100):
        print(i)
        batch_images = imageDataset[i*100:i*100+100]
        #list_backbone_structures = [utils.keep_backbone(parser.get_structure("A", dataset_path + "predicted_test_" + str(j) + ".pdb"))[None, :, :]
        #                   for j in range(i*100, i*100+100)]
        #batch_backbone_structures = torch.tensor(np.concatenate(list_backbone_structures, axis=0), dtype=torch.float32)
        batch_rotation_matrices = all_rotation_matrices[i*100:i*100+100, :, :]
        #start = time.time()
        batch_backbone_structures, _, _, _, _, _ = model(torch.ones((100, 1))*100000, batch_images)
        print("epoch:", epoch)
        print(i/100)
        print("Deformed")
        #print("Backbone shape", batch_backbone_structures.shape)
        ## We then rotate the structure and project them on the x-y plane.
        deformed_images = renderer.compute_x_y_values_all_atoms(batch_backbone_structures, batch_rotation_matrices)
        print("\n\n")
        #plt.imshow(deformed_images[0], cmap="gray")
        #plt.show()
        all_images_predicted.append(deformed_images)

        all_images_predicted_torch = torch.concat(all_images_predicted, dim=0)
        torch.save(all_images_predicted_torch,
                   "../VAEProtein/data/vaeContinuousMD6DomainsDecoupledLatent/predicted_structures/predictedDataSet")

"""
path_predicted_structures = "../VAEProtein/data/vaeContinuousMD_open/predicted_structures/"
path_pose_matrix = "../VAEProtein/data/vaeContinuousMD_open/training_rotations_matrices"
training_rotation_matrices = torch.load(path_pose_matrix)
parser = PDBParser(PERMISSIVE=0)
for i in range(10000):
    print(i)
    structure = torch.tensor(utils.keep_backbone(parser.get_structure("A", path_predicted_structures + "predicted_test_" + str(i) + ".pdb")),
                             dtype=torch.float32)[None, :, :]

    image = renderer.compute_x_y_values_all_atoms(structure, training_rotation_matrices[i:i+1, :, :])
    plt.imshow(image[0, :, :].detach().numpy())
    plt.show()

all_images_predicted_torch = torch.concat(all_images_predicted, dim=0)
print(all_images_predicted.shape)
torch.save(all_images_predicted_torch, "../VAEProtein/data/vaeContinuousMD6DomainsDecoupledLatent/predicted_structures/predictedDataSet")
