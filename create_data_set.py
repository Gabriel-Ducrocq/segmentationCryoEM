import numpy as np
import torch
import utils
from scipy.spatial.transform import Rotation
import yaml
from imageRenderer import Renderer



def generate_global_rotation_axis(N):
    rotation_axis = np.random.normal(size=(N, 3))
    norm = np.sqrt(np.sum(rotation_axis**2, axis=1))
    return rotation_axis/norm[:, None]

def generate_global_rotation_angle(N):
    return np.random.uniform(0, 2*np.pi, size=(N, 1))

def from_axis_angle_to_matrix(axis_angle):
    rotations = Rotation.from_rotvec(axis_angle)
    return rotations.as_matrix()

def extractor_domain(conformation, type):
    assert type in ["translation", "rotation_axis", "rotation_angle"]
    dim = 1 if type == "rotation_angle" else 3
    N = conformation["N_sample"]
    domains = conformation["domains"]
    N_domains = len(domains)
    per_domain = tuple(map(lambda domain: domain[type], domains.values()))
    per_domain = np.reshape(per_domain, (N_domains, dim))
    return np.repeat(per_domain[None, :, :], N, axis=0)


with open("dataset.yaml", "r") as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
print("Device", device)
print(config["device"] == "cuda")
print(torch.cuda.is_available())
pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device)

print("start")
N_domains = len(config["conformations"]["conformation1"]["domains"])
translation_data_set = np.vstack(tuple(map(lambda x: extractor_domain(x, "translation"), config["conformations"].values())))
rotation_axis_dataset = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_axis"), config["conformations"].values())))
rotation_angle_dataset = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_angle"), config["conformations"].values())))
axis_angle_conformations_dataset = rotation_axis_dataset*rotation_angle_dataset
total_N_sample = axis_angle_conformations_dataset.shape[0]
global_rotation_axis = generate_global_rotation_axis(total_N_sample)
global_rotation_angle = generate_global_rotation_angle(total_N_sample)
global_axis_angle_dataset = global_rotation_axis*global_rotation_angle
global_rotation_matrix_dataset = from_axis_angle_to_matrix(global_axis_angle_dataset)
conformation_matrix_dataset = tuple(map(from_axis_angle_to_matrix, axis_angle_conformations_dataset))
conformation_matrix_dataset = np.reshape(conformation_matrix_dataset, (total_N_sample, N_domains, 3, 3))

print("Done having the matrices !")


features = np.load(config["protein_features"], allow_pickle=True)
features = features.item()
local_frame = features["local_frame"]
absolute_positions = features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0)
relative_positions = np.matmul(absolute_positions, local_frame)

cutoff1 = config["conformations"]["conformation1"]["domains"]["domain1"]["end"]
cutoff2 = config["conformations"]["conformation1"]["domains"]["domain3"]["end"]

absolute_positions = torch.tensor(absolute_positions, dtype=torch.float32, device=device)
translation_data_set = torch.tensor(translation_data_set, dtype=torch.float32, device=device)
conformation_matrix_dataset = torch.tensor(conformation_matrix_dataset, dtype=torch.float32, device=device)
global_rotation_matrix_dataset = torch.tensor(global_rotation_matrix_dataset, dtype=torch.float32, device=device)
local_frame = torch.tensor(local_frame, dtype=torch.float32, device=device)
relative_positions = torch.tensor(relative_positions, dtype=torch.float32, device=device)

deformed_structures = utils.deform_structure(absolute_positions, cutoff1, cutoff2, translation_data_set,
                                             conformation_matrix_dataset, local_frame, relative_positions,
                                             1510, device)


print("Rest to create images !")
deformed_images = renderer.compute_x_y_values_all_atoms(deformed_structures, global_rotation_matrix_dataset)

