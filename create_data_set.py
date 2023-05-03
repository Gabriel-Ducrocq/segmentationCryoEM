import numpy as np
import torch
import utils
from scipy.spatial.transform import Rotation
import yaml
from imageRenderer import Renderer
import matplotlib.pyplot as plt
from protein.main import rotate_domain_pdb_structure_matrix
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO



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

def extractor_domain(conformation, type):
    """

    :param conformation: dictionnary describing the characteristics of the conformation, e.g number of samples,
    domain definitions...
    :param type: str describing the quantity we look for (translation, rot axis or rot angle)
    :return:
    """
    assert type in ["translation", "rotation_axis", "rotation_angle", "rotation_angle_min", "rotation_angle_max", "step"]
    dim = 1 if type in ["rotation_angle", "rotation_angle_min", "rotation_angle_max", "step"] else 3
    N = conformation["N_sample"]
    domains = conformation["domains"]
    N_domains = len(domains)
    ## For each domain we extract the transformation
    per_domain = tuple(map(lambda domain: domain[type], domains.values()))
    ## We get it to the format (N_domain, dim)
    per_domain = np.reshape(per_domain, (N_domains, dim))
    ## We repeat this pattern for the number of sample of the conformation
    return np.repeat(per_domain[None, :, :], N, axis=0)


with open("datasetContinuous.yaml", "r") as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
print("Device", device)
print(config["device"] == "cuda")
print(torch.cuda.is_available())
pixels_x = np.linspace(-105, 105, num=64).reshape(1, -1)
pixels_y = np.linspace(-105, 105, num=64).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device)

print("start")

def get_structures(config, device="cpu"):
    """
    Deform the base structure using the conformations described in the yaml file config
    :param config: dictionnary obtained reading a yaml file, describing the possible conformations
    :return: torch tensor (N_total, N_atoms, 3) describing to the conformationnally displaced structures. N_total is the
    sum of the number of particles for each conformation
    """
    #Getting the number of domains for conformation 1
    N_domains = len(config["conformations"]["conformation1"]["domains"])
    # We get the translation data per domain for all conformations
    translation_data_set = np.vstack(tuple(map(lambda x: extractor_domain(x, "translation"), config["conformations"].values())))
    # We get the rotation axis per domain for all conformations
    rotation_axis_dataset = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_axis"), config["conformations"].values())))
    # We get the rotation angle per domain for all conformations
    if config["type"] == "continuous":
        rotation_angle_min = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_angle_min"), config["conformations"].values())))
        rotation_angle_max = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_angle_max"), config["conformations"].values())))
        rotation_angle_dataset = np.random.uniform(low=rotation_angle_min, high=rotation_angle_max)
    elif config["type"] == "continuous_zhong":
        ##If Ellen Zhong type, we sample conformations regularly and generate several thousands images with one conformation.
        ## See cryoFold or cryoDRGN
        rotation_angle_min = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_angle_min"), config["conformations"].values())))
        rotation_angle_max = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_angle_max"), config["conformations"].values())))
        step = np.vstack(tuple(map(lambda x: extractor_domain(x, "step"), config["conformations"].values())))
        N = config["conformations"]["conformation1"]["N_sample"]
        all_particle_dataset = []
        for n_particle in range(N):
            all_domains_dataset = []
            for rot_min_part, st_part, rot_max_part in zip(rotation_angle_min[n_particle], step[n_particle],rotation_angle_max[n_particle]):
                rotation_angle_dataset = []
                for rot_min, st, rot_max in zip(rot_min_part, st_part, rot_max_part):
                    if st != 0:
                        rotation_angle_dataset.append(np.linspace(rot_min,rot_max, st)[n_particle])
                    else:
                        assert step == N, "Number of steps not equal to number of particles set in the cryoDRGN style dataset"
                        rotation_angle_dataset.append(np.zeros(50))

                all_domains_dataset.append(rotation_angle_dataset)

            all_particle_dataset.append(all_domains_dataset)

        rotation_angle_dataset = np.array(all_particle_dataset)[:, :, :]
    else:
        rotation_angle_dataset = np.vstack(tuple(map(lambda x: extractor_domain(x, "rotation_angle"), config["conformations"].values())))

    # We get the axis_angle representation for each domain.
    axis_angle_conformations_dataset = rotation_axis_dataset * rotation_angle_dataset
    # We get the number of samples of the conformation.
    total_N_sample = axis_angle_conformations_dataset.shape[0]
    #We turn the rotation per domain from axis_angle representation to matrix representation
    conformation_matrix_dataset = tuple(map(from_axis_angle_to_matrix, axis_angle_conformations_dataset))
    conformation_matrix_dataset = np.reshape(conformation_matrix_dataset, (total_N_sample, N_domains, 3, 3))
    features = np.load(config["protein_features"], allow_pickle=True)
    features = features.item()
    local_frame = features["local_frame"]
    local_frame_in_columns = local_frame.T
    if config["type"] == "continuous_zhong":
        for i in range(total_N_sample):
            if i%10 == 0:
                print(i)

            pdb_parser = PDBParser()
            io = PDBIO()
            struct = pdb_parser.get_structure("A", "data/vaeContinuous/ranked_0.pdb")
            rotate_domain_pdb_structure_matrix(struct, 1353, 1510, conformation_matrix_dataset[i, 2, :, :], local_frame_in_columns)
            io.set_structure(struct)
            io.save("data/true_structure"+str(i)+".pdb", preserve_atom_numbering = True)

    #We get the relative positions of each atom in the local frame given by the first residue.
    features = np.load(config["protein_features"], allow_pickle=True)
    features = features.item()
    local_frame = features["local_frame"]
    absolute_positions = features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0)
    relative_positions = np.matmul(absolute_positions, local_frame)

    cutoff1 = config["conformations"]["conformation1"]["domains"]["domain1"]["end"]
    cutoff2 = config["conformations"]["conformation1"]["domains"]["domain3"]["end"]

    absolute_positions = torch.tensor(absolute_positions, dtype=torch.float32, device=device)
    conformation_matrix_dataset = torch.tensor(conformation_matrix_dataset, dtype=torch.float32, device=device)
    translation_data_set = torch.tensor(translation_data_set, dtype=torch.float32, device=device)
    local_frame = torch.tensor(local_frame, dtype=torch.float32, device=device)
    relative_positions = torch.tensor(relative_positions, dtype=torch.float32, device=device)
    #We actually deform the structure
    deformed_structures = utils.deform_structure(absolute_positions, cutoff1, cutoff2, translation_data_set,
                                                 conformation_matrix_dataset, local_frame, relative_positions,
                                                 1510, device)

    return deformed_structures


def get_images(deformed_structures, noise_variance, config, device="cpu"):
    """
    Gives a random orientatin to each deformed structure and projects it into a 2d image.
    :param deformed_structures: torch tensor (N_total, N_atoms, 3) describing to the conformationnally displaced structures. N_total is the
    sum of the number of particles for each conformation
    :param noise_variance: float32, describes the variance of the noise added to the pictures
    :return: torch tensor (N_total, N_pix_x, N_pix_y)
    """
    total_N_sample = deformed_structures.shape[0]
    # We generate pose rotation axis
    global_rotation_axis = generate_global_rotation_axis(total_N_sample)
    # We generate pose rotation angle
    global_rotation_angle = generate_global_rotation_angle(total_N_sample)
    # We get the axis_angle representation of the poses
    global_axis_angle_dataset = global_rotation_axis * global_rotation_angle
    # We turn the poses into a rotation matrix
    global_rotation_matrix_dataset = from_axis_angle_to_matrix(global_axis_angle_dataset)
    global_rotation_matrix_dataset = torch.tensor(global_rotation_matrix_dataset, dtype=torch.float32, device=device)
    all_deformed_images = torch.empty((total_N_sample, 64, 64))
    for i in range(0, 20):
        print(i)
        deformed_images = renderer.compute_x_y_values_all_atoms(deformed_structures[i*500:(i+1)*500], global_rotation_matrix_dataset[i*500:(i+1)*500])
        #deformed_images += torch.randn_like(deformed_images)*np.sqrt(noise_variance)
        all_deformed_images[i*500:(i+1)*500] = deformed_images

    return all_deformed_images, global_rotation_matrix_dataset


deformed_structures = get_structures(config)
deformed_images, global_rotation_axis = get_images(deformed_structures, 0.2, None)
noisy_images = deformed_images #+ torch.randn_like(deformed_images[0])*np.sqrt(0.2)

MSD = torch.sum((deformed_images - noisy_images)**2, dim=(-2,-1))
print("MSD:", MSD)
print(torch.var(deformed_images))
print(torch.mean(deformed_images))
torch.save(deformed_images, "data/vaeContinuousNoisyZhongStyleNoNoise/continuousConformationDataSet")
torch.save(global_rotation_axis, "data/vaeContinuousNoisyZhongStyleNoNoise/rotationPoseDataSet")

for i in range(1000):
    plt.imshow(deformed_images[i], cmap="gray")
    plt.show()