import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
import utils
import scipy
import yaml



def construct_conformation_translation(conformation):
    N = conformation["N_sample"]
    domains = conformation["domains"]
    N_domains = len(domains)
    all_translations = map(lambda domain: domain["translation"],  domains.values())
    print(np.reshape(all_translations), (N_domains, 3))


with open("dataset.yaml", "r") as file:
    config = yaml.safe_load(file)

construct_conformation_translation(config["conformations"]["conformation1"])
device = torch.device("cuda" if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")


features = np.load(config["protein_features"], allow_pickle=True)
features = features.item()
absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
absolute_positions = absolute_positions.to(device)

local_frame = torch.tensor(features["local_frame"])
local_frame = local_frame.to(device)

relative_positions = torch.matmul(absolute_positions, local_frame)
conformation1 = torch.tensor(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), dtype=torch.float32)
conformation2 = torch.tensor(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=torch.float32)
conformation1_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
conformation1_rotation_angle = torch.tensor(np.array([np.pi / 4, 0, np.pi / 8, 0]), dtype=torch.float32)
conformation1_rotation_axis_angle = conformation1_rotation_axis * conformation1_rotation_angle[:, None]
conformation1_rotation_matrix = axis_angle_to_matrix(conformation1_rotation_axis_angle)
conformation2_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
conformation2_rotation_angle = torch.tensor(np.array([-np.pi / 4, 0, 0, 0]), dtype=torch.float32)
conformation2_rotation_axis_angle = conformation2_rotation_axis * conformation2_rotation_angle[:, None]
conformation2_rotation_matrix = axis_angle_to_matrix(conformation2_rotation_axis_angle)

conformation1_rotation_matrix = torch.broadcast_to(conformation1_rotation_matrix, (5, 4, 3, 3))
conformation2_rotation_matrix = torch.broadcast_to(conformation2_rotation_matrix, (5, 4, 3, 3))
conformation_rotation_matrix = torch.cat([conformation1_rotation_matrix, conformation2_rotation_matrix], dim=0)
conformation1 = torch.broadcast_to(conformation1, (5, 12))
conformation2 = torch.broadcast_to(conformation2, (5, 12))
true_deformations = torch.cat([conformation1, conformation2], dim=0)
#rotation_angles = torch.tensor(np.random.uniform(0, 2 * np.pi, size=(10, 1)), dtype=torch.float32, device=device)
rotation_angles = torch.zeros((10, 1), dtype=torch.float32, device=device)
rotation_axis = torch.randn(size=(10, 3), device=device)
rotation_axis = rotation_axis / torch.sqrt(torch.sum(rotation_axis ** 2, dim=1))[:, None]
axis_angle_format = rotation_axis * rotation_angles
rotation_matrices = axis_angle_to_matrix(axis_angle_format)


batch_data_for_deform = torch.reshape(true_deformations, (batch_size, N_input_domains, 3))
batch_conformation_rotation_matrices = conformation_rotation_matrix
deformed_structures, base_structure = utils.deform_structure(absolute_positions, domain_cutoff[1], domain_cutoff[2], batch_data_for_deform,
                                             batch_conformation_rotation_matrices, local_frame, relative_positions,
                                             1510, device)

l = 2
start_residue_domain = domain_cutoff[l]
end_residue_domain = domain_cutoff[l + 1]
opt_rotation, _ = scipy.spatial.transform.Rotation.align_vectors(absolute_positions[3 * start_residue_domain:3 * end_residue_domain],
    deformed_structures[0, 3 * start_residue_domain:3 * end_residue_domain, :])

axis_angle = opt_rotation.as_rotvec()
print(axis_angle)
norm = np.sqrt(np.sum(axis_angle**2))
print(norm)
print(axis_angle/norm)

print(absolute_positions[3 * start_residue_domain:3 * end_residue_domain] - deformed_structures[0][3 * start_residue_domain:3 * end_residue_domain])