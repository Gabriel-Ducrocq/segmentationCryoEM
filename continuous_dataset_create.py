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
dataset_path = "../VAEProtein/data/MD_dataset/"
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
renderer = Renderer(pixels_x, pixels_y, std=1, device=device, use_ctf=True, N_heavy=3018)

"""
relative_positions = torch.matmul(absolute_positions, local_frame)
conformation1 = torch.tensor(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), dtype=torch.float32)
conformation2 = torch.tensor(np.array([0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=torch.float32)
conformation1_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
#conformation1_rotation_angle = torch.tensor(np.array([-np.pi / 4, 0, np.pi/2, 0]), dtype=torch.float32)
conformation1_rotation_angle = torch.zeros((50000, 4), dtype=torch.float32)
#conformation1_rotation_angle[:, 2] = -torch.rand(size=(5000,))*torch.pi
conformation1_rotation_angle[:, 2] = torch.randn(size=(50000,))*0.1 -np.pi/3
#conformation1_rotation_axis_angle = conformation1_rotation_axis*conformation1_rotation_angle[:, None]
conformation1_rotation_axis_angle = torch.broadcast_to(conformation1_rotation_axis[None, :, :], (50000, 4, 3))\
                                    * conformation1_rotation_angle[:, :, None]

conformation1_rotation_matrix = axis_angle_to_matrix(conformation1_rotation_axis_angle)

#conformation2_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
#conformation2_rotation_angle = torch.tensor(np.array([0, 0, -np.pi/2, 0]), dtype=torch.float32)
conformation2_rotation_angle = torch.zeros((50000, 4), dtype=torch.float32)
#conformation2_rotation_angle[:, 2] = -torch.rand(size=(5000,))*torch.pi
conformation2_rotation_angle[:, 2] = torch.randn(size=(50000,))*0.1 -2*np.pi/3
#conformation2_rotation_axis_angle = conformation2_rotation_axis * conformation2_rotation_angle[:, None]
conformation2_rotation_axis_angle = torch.broadcast_to(conformation1_rotation_axis[None, :, :], (50000, 4, 3))\
                                    * conformation2_rotation_angle[:, :, None]
conformation2_rotation_matrix = axis_angle_to_matrix(conformation2_rotation_axis_angle)

#conformation1_rotation_matrix = torch.broadcast_to(conformation1_rotation_matrix, (5000, 4, 3, 3))
#conformation2_rotation_matrix = torch.broadcast_to(conformation2_rotation_matrix, (5000, 4, 3, 3))
conformation_rotation_matrix = torch.cat([conformation1_rotation_matrix, conformation2_rotation_matrix], dim=0)
conformation1 = torch.broadcast_to(conformation1, (50000, 12))
conformation2 = torch.broadcast_to(conformation2, (50000, 12))
true_deformations = torch.cat([conformation1, conformation2], dim=0)
rotation_angles = torch.tensor(np.random.uniform(0, 2*np.pi, size=(100000,1)), dtype=torch.float32, device=device)
#rotation_angles = torch.tensor(np.random.uniform(0, 2*np.pi, size=(10000)), dtype=torch.float32, device=device)
rotation_axis = torch.randn(size=(100000, 3), device=device)
rotation_axis = rotation_axis/torch.sqrt(torch.sum(rotation_axis**2, dim=1))[:, None]
axis_angle_format = rotation_axis*rotation_angles
rotation_matrices = axis_angle_to_matrix(axis_angle_format)

training_set = true_deformations.to(device)
#rotation_matrices = torch.transpose(rotation_matrices, dim0=-2, dim1=-1)
training_rotations_matrices = rotation_matrices.to(device)
training_rotations_angles = rotation_angles.to(device)
training_rotations_axis = rotation_axis.to(device)
#conformation_rotation_matrix = torch.transpose(conformation_rotation_matrix, dim0=-2, dim1=-1)
training_conformation_rotation_matrix = conformation_rotation_matrix.to(device)
print("PATH")
print(dataset_path)
torch.save(training_set, dataset_path + "training_set")
torch.save(training_rotations_angles, dataset_path + "training_rotations_angles")
torch.save(training_rotations_axis, dataset_path + "training_rotations_axis")
torch.save(training_rotations_matrices, dataset_path + "training_rotations_matrices")
torch.save(training_conformation_rotation_matrix, dataset_path + "training_conformation_rotation_matrices")
#torch.save(test_set, dataset_path + "test_set.npy")

training_set = torch.load(dataset_path + "training_set").to(device)
training_rotations_angles = torch.load(dataset_path + "training_rotations_angles").to(device)
training_rotations_axis = torch.load(dataset_path + "training_rotations_axis").to(device)
training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices").to(device)
training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices")
print("Creating dataset")
print("Done creating dataset")
"""
training_indexes = torch.tensor(np.array(range(N_images)))
all_images = []
all_folders = [int(fold) for fold in os.listdir(dataset_path) if fold != ".DS_Store"]
all_folders = sorted(all_folders)
all_folders = [str(fold) for fold in all_folders]
parser = PDBParser(PERMISSIVE=0)
batch_rotation_matrices = torch.eye(3)[None, :, :].repeat(10, 1, 1)
all_rotation_matrices = torch.zeros((10000, 3, 3), dtype=torch.float32)
for epoch in range(0,1):
    data_loader = iter(DataLoader(training_indexes, batch_size=batch_size, shuffle=False))
    print(all_folders)
    for i,dir in enumerate(all_folders):
        print(dir)
        list_structures = [dataset_path + dir + "/" + struct for struct in sorted(os.listdir(dataset_path + dir)) if ".pdb" in struct]
        angle_rotation = torch.tensor(np.load(dataset_path + dir + "/angle_rotation.npy"), dtype=torch.float32)
        axis_rotation = torch.tensor(np.load(dataset_path + dir + "/axis_rotation.npy"), dtype=torch.float32)
        all_rotation_matrices[i*10:i*10+10, :, :] = axis_angle_to_matrix(axis_rotation*angle_rotation[:, None])
        start = time.time()
        print("epoch:", epoch)
        print(i/1000)
        #batch_data = next(iter(data_loader))
        batch_indexes = next(iter(data_loader))
        ##Getting the batch translations, rotations and corresponding rotation matrices
        #batch_data = training_set[batch_indexes]
        #batch_rotations_angles = training_rotations_angles[batch_indexes]
        #batch_rotations_axis = training_rotations_axis[batch_indexes]
        #batch_rotation_matrices = training_rotations_matrices[batch_indexes]
        #batch_data_for_deform = torch.reshape(batch_data, (batch_size, N_input_domains, 3))
        #batch_conformation_rotation_matrices = training_conformation_rotation_matrix[batch_indexes]
        ## Deforming the structure for each batch data point
        #deformed_structures = utils.deform_structure(absolute_positions, cutoff1, cutoff2,batch_data_for_deform,
        #                                             batch_conformation_rotation_matrices, local_frame, relative_positions,
        #                                             1510, device)


        print("Deformed")
        list_backbone_structures = [utils.keep_backbone(parser.get_structure("A", file))[None, :, :] for file in list_structures]
        batch_backbone_structures = torch.tensor(np.concatenate(list_backbone_structures, axis=0), dtype=torch.float32)
        print("Backbone shape", batch_backbone_structures.shape)
        print("Rot mat:", batch_rotation_matrices.shape)
        ## We then rotate the structure and project them on the x-y plane.
        deformed_images = renderer.compute_x_y_values_all_atoms(batch_backbone_structures, batch_rotation_matrices)
        print(torch.mean(torch.var(deformed_images, dim=(1, 2))))
        deformed_images += torch.randn_like(deformed_images)*np.sqrt(noise_var)
        print(torch.mean(torch.var(deformed_images, dim=(1,2))))
        print("\n\n")
        #plt.imshow(deformed_images[0], cmap="gray")
        #plt.show()
        all_images.append(deformed_images)


all_images = torch.concat(all_images, dim=0)
print(all_images.shape)
torch.save(all_images, "../VAEProtein/data/vaeTwoClustersMDLatent40/" + "continuousConformationDataSet")
torch.save(all_rotation_matrices, "../VAEProtein/data/vaeTwoClustersMDLatent40/training_rotations_matrices")