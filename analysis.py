import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import utils
from imageRenderer import Renderer
from Bio.PDB.PDBParser import PDBParser
from pytorch3d.transforms import quaternion_to_axis_angle
from protein.main import rotate_residues
from protein.main import translate_residues
from Bio.PDB.PDBIO import PDBIO
import Bio.PDB as bpdb

class ResSelect(bpdb.Select):
    def accept_residue(self, res):
        if res.get_resname() == "LBV":
            return False
        else:
            return True

#dataset_path="data/vaeContinuousCTFNoisyBiModalAngle100kEncoder/"
dataset_path="/Users/gabdu45/PycharmProjects/VAEProtein/data/vaeTwoClustersMDLatent40/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
#This represent the number of true domains
N_domains = 6
N_pixels = 140*140
#This represents the number of domain we think there are
N_input_domains = 6
latent_dim = 10
num_nodes = 1006
cutoff1 = 300
cutoff2 = 1353
K_nearest_neighbors = 30
one_latent_per_domain = False
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

model_path = "/Users/gabdu45/PycharmProjects/VAEProtein/data/vaeTwoClustersMDLatent40/full_model2119"
model = torch.load(model_path, map_location=torch.device(device))


#training_set = torch.load(dataset_path + "training_set", map_location=torch.device(device)).to(device)
#training_rotations_angles = torch.load(dataset_path + "training_rotations_angles", map_location=torch.device(device)).to(device)
#training_rotations_axis = torch.load(dataset_path + "training_rotations_axis", map_location=torch.device(device)).to(device)
training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices", map_location=torch.device(device)).to(device)
#training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices", map_location=torch.device(device))
images = torch.load(dataset_path + "continuousConformationDataSet", map_location=torch.device(device))
#print("SHOULD WE USE ENCODER:", model.use_encoder)
#print("DATASET SIZE:", training_set.shape)
training_indexes = torch.tensor(np.array(range(10000)))
all_latent_distrib = []
all_indexes = []
all_rot = []
model.device = "cpu"
all_translations = []
all_rotations_per_residues = []
all_translations_per_residues = []

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
        batch_images = torch.flatten(images[batch_indexes], start_dim=1, end_dim=2)
        latent_distrib = model.encoder.forward(batch_images)
        if not one_latent_per_domain:
            transforms = model.decoder.forward(latent_distrib[:, :latent_dim])
            transforms = torch.reshape(transforms, (batch_size, N_input_domains, 2 * 3))
        else:
            output_all_domains = []
            for k in range(N_input_domains):
                latent_variable_k = latent_distrib[:, latent_dim*k:latent_dim*(k+1)]
                output_k = model.decoder.forward(latent_variable_k)
                output_all_domains.append(output_k[:, None, :])

            transforms = torch.concat(output_all_domains, dim=1)

        scalars_per_domain = transforms[:, :, :3]

        ones = torch.ones(size=(batch_size, N_input_domains, 1), device=device)
        quaternions_per_domain = torch.cat([ones, transforms[:, :, 3:]], dim=-1)
        rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions_per_domain)
        translation_vectors = torch.matmul(scalars_per_domain, local_frame)

        weights = model.compute_mask()
        rotations_per_residue = model.compute_rotations(quaternions_per_domain, weights)
        all_rotations_per_residues.append(rotations_per_residue.detach().numpy())
        translation_per_residue = torch.matmul(weights, translation_vectors)
        all_translations_per_residues.append(translation_per_residue.detach().numpy())

        all_rot.append(rotation_per_domains_axis_angle.detach().cpu().numpy())
        all_translations.append(translation_vectors.detach().cpu().numpy())
        all_latent_distrib.append(latent_distrib.detach().cpu().numpy())
        all_indexes.append(batch_indexes.detach().cpu().numpy())


np.save(dataset_path + "latent_distrib.npy", np.concatenate(all_latent_distrib))
np.save(dataset_path + "indexes.npy", np.concatenate(all_indexes))

np.save(dataset_path + "all_rotations.npy", np.concatenate(all_rot))
np.save(dataset_path + "all_translations.npy", np.concatenate(all_translations))
np.save(dataset_path + "all_rotations_per_residue.npy", np.concatenate(all_rotations_per_residues, axis=0))
np.save(dataset_path + "all_translations_per_residue.npy", np.concatenate(all_translations_per_residues, axis=0))


rotations_per_domain = np.load(dataset_path + "all_rotations.npy")
translations_per_domain = np.load(dataset_path + "all_translations.npy")

all_rotations_per_residues = np.load(dataset_path + "all_rotations_per_residue.npy")
all_translations_per_residues = np.load(dataset_path + "all_translations_per_residue.npy")


pdb_path = "../VAEProtein/data/MD_dataset/"
saving_path = "/Users/gabdu45/PycharmProjects/VAEProtein/data/vaeTwoClustersMDLatent40/predicted_structures/"

for i in range(0, 10001):
    print(i)
    parser = PDBParser(PERMISSIVE=0)
    translations = all_translations_per_residues[i]
    rotations = all_rotations_per_residues[i]
    structure = parser.get_structure("A", pdb_path + "test_1.pdb")
    io = bpdb.PDBIO()
    io.set_structure(structure)
    io.save(saving_path + "predicted_test_"+ str(i)+ ".pdb", ResSelect())
    structure = parser.get_structure("A", saving_path + "predicted_test_" + str(i) + ".pdb")
    translate_residues(structure, translations)
    rotate_residues(structure, rotations, local_frame)
    io = PDBIO()
    io.set_structure(structure)
    io.save(saving_path + "predicted_test_"+ str(i)+ ".pdb")

