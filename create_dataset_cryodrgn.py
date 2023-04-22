import numpy as np
import torch
import utils
from scipy.spatial.transform import Rotation
import yaml
from imageRenderer import Renderer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from protein.main import rotate_domain_pdb_structure_matrix
import imageRenderer

device = "cpu"
N_images_per_conf = 1000
N_conf = 10
with open("datasetContinuousZhongFashion.yaml", "r") as file:
    config = yaml.safe_load(file)


def generate_global_rotation_axis(N):
    """
    Create uniform pose rotation
    :param N: integer, number of rotation axis to create
    :return: np.array of size (N, 3) of rotation axis
    """
    rotation_axis = np.random.normal(size=(N, 3))
    norm = np.sqrt(np.sum(rotation_axis**2, axis=1))
    return rotation_axis/norm[:, None]

def generate_global_rotation_angle(N):
    """

    :param N: integer, number of rotation axis to create
    :return: np.array of size (N, 1) of rotation axis
    """
    return np.random.uniform(0, 2*np.pi, size=(N, 1))

def from_axis_angle_to_matrix(axis_angle):
    rotations = Rotation.from_rotvec(axis_angle)
    return rotations.as_matrix()

def from_matrix_to_axis_angle(matrix):
    rotations = Rotation.from_matrix(matrix)
    return rotations.as_rotvec()


#Get regular spacing along the 1D continuous motion
rotation_axis = np.repeat(np.array([[0, 1, 0]]), 10, axis=0)
rotation_angle = np.arange(-2.5, 0, 0.25)[:, None]
#Get in axis angle format
axis_angle = rotation_axis * rotation_angle
#Get in rotation matrix format
conformation_matrix_dataset = tuple(map(from_axis_angle_to_matrix, axis_angle))
conformation_matrix_dataset = np.reshape(conformation_matrix_dataset, (10, 3, 3))

features = np.load(config["protein_features"], allow_pickle=True)
features = features.item()
local_frame = features["local_frame"]
local_frame_in_columns = torch.tensor(local_frame.T, dtype=torch.float32)
#for i in range(50):
#    if i % 10 == 0:
#        print(i)


#    pdb_parser = PDBParser()
#    io = PDBIO()
#    struct = pdb_parser.get_structure("A", "data/vaeContinuous/ranked_0.pdb")
#    rotate_domain_pdb_structure_matrix(struct, 1353, 1510, conformation_matrix_dataset[i, :, :],
#                                       local_frame_in_columns)
#    io.set_structure(struct)
#    io.save("data/true_structure" + str(i) + ".pdb", preserve_atom_numbering=True)

# Sampling the conformations present in the dataset among all the 50 conformations
#random_conf = []
#for i in range(N_conf):
#    if np.random.uniform() < 1/2:
#        random_conf.append(np.round(np.random.uniform(0, 10)))
#    else:
#        random_conf.append(np.round(np.random.uniform(40, 49)))

#random_conf = [ 9, 6, 48, 2, 44, 41, 4, 47, 1, 45]
random_conf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
conformation_matrix_rot_dataset = conformation_matrix_dataset[np.array(random_conf, dtype=int)]
conformation_matrix_rot_dataset = np.stack([np.zeros((N_conf, 3, 3)), np.zeros((N_conf, 3, 3)), conformation_matrix_rot_dataset[:, :, :]], axis=1)
conformation_matrix_rot_dataset[:, :2, 0, 0] = conformation_matrix_rot_dataset[:, :2, 1, 1] = conformation_matrix_rot_dataset[:, :2, 2, 2] =1
print("random conf", np.array(random_conf, dtype=int))
features = np.load("data/features.npy", allow_pickle=True)
features = features.item()
local_frame = features["local_frame"]
absolute_positions = features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0)
relative_positions = np.matmul(absolute_positions, local_frame)

cutoff1 = config["conformations"]["conformation1"]["domains"]["domain1"]["end"]
cutoff2 = config["conformations"]["conformation1"]["domains"]["domain3"]["start"]



absolute_positions = torch.tensor(absolute_positions, dtype=torch.float32, device=device)
#conformation_matrix_dataset = torch.tensor(conformation_matrix_dataset, dtype=torch.float32, device=device)

#Setting translation to 0.
translation_data_set = np.zeros((10, 3, 3))
translation_data_set = torch.tensor(translation_data_set, dtype=torch.float32, device=device)
local_frame = torch.tensor(local_frame, dtype=torch.float32, device=device)
relative_positions = torch.tensor(relative_positions, dtype=torch.float32, device=device)

conformation_matrix_rot_dataset = torch.tensor(conformation_matrix_rot_dataset, dtype=torch.float32, device=device)
#We actually deform the structure

print("CUTOFF1", cutoff1, cutoff2)
print(conformation_matrix_rot_dataset.shape)
print("REL POS", relative_positions.shape)
print(translation_data_set)
deformed_structures = utils.deform_structure(absolute_positions, cutoff1, cutoff2, translation_data_set,
                                             conformation_matrix_rot_dataset, local_frame, relative_positions,
                                             1510, device)

print("DEFORMED",torch.max(torch.abs(deformed_structures)))
print("Relative",torch.max(torch.abs(relative_positions)))
print("absolue",torch.max(torch.abs(absolute_positions)))
print("local_frame",torch.max(torch.abs(local_frame)))
print("rot conf",torch.max(torch.abs(conformation_matrix_rot_dataset)))
#print("\n")
#print(deformed_structures[:, cutoff2:, :])


print("DEFORMED")
pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device)

noise_variance = 0.2
all_images = []
all_pose_rotation_matrix = []
for n in range(N_conf):
    print("N_conf", n)
    global_rotation_axis = generate_global_rotation_axis(N_images_per_conf)
    # We generate pose rotation angle
    global_rotation_angle = generate_global_rotation_angle(N_images_per_conf)
    global_axis_angle_dataset = global_rotation_axis * global_rotation_angle
    #print("cutoff1", cutoff1)
    #print("cutoff2", cutoff2)
    #print("glob axis angle", global_axis_angle_dataset.shape)
    # We turn the poses into a rotation matrix
    global_rotation_matrix_dataset = from_axis_angle_to_matrix(global_axis_angle_dataset)
    global_rotation_matrix_dataset = torch.tensor(global_rotation_matrix_dataset, dtype=torch.float32, device=device)
    #print("globa rot", global_rotation_matrix_dataset.shape)
    all_pose_rotation_matrix.append(global_rotation_matrix_dataset)
    #all_deformed_images = torch.empty((N_images_per_conf, 64, 64))
    deformed_structure = deformed_structures[n][None, :, :]
    deformed_structure = deformed_structure.repeat(N_images_per_conf, 1, 1)
    #print("structures:", deformed_structure.shape)
    #print(deformed_structure)
    #print(torch.max(torch.abs(deformed_structure)))
    deformed_images = renderer.compute_x_y_values_all_atoms(deformed_structure,
                                                            global_rotation_matrix_dataset)
    print("Images:", deformed_images.shape)
    print(deformed_images)
    deformed_images += torch.randn_like(deformed_images) * np.sqrt(noise_variance)
    all_images.append(deformed_images)
    #all_deformed_images = deformed_images

all_images = torch.concat(all_images)
all_pose_rotation_matrix = torch.concat(all_pose_rotation_matrix)
print(all_images.shape)
print(all_pose_rotation_matrix.shape)
torch.save(all_images, "data/vaeContinuousNoisyZhongStyle2/continuousConformationDataSet")
torch.save(all_pose_rotation_matrix, "data/vaeContinuousNoisyZhongStyle2/rotationPoseDataSet")
np.save("data/vaeContinuousNoisyZhongStyle2/structures_indexes.npy", np.array(random_conf, dtype=int))
