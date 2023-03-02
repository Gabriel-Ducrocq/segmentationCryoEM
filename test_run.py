import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import utils
from imageRenderer import Renderer
from pytorch3d.transforms import axis_angle_to_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "data/vae2Conformations/full_model"
network = torch.load(model_path, map_location=torch.device(device))
network.device = "cpu"
dataset_path="data/vae2Conformations/test_run/"
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
model_path = "data/vae2Conformations/full_model"
model = torch.load(model_path, map_location=torch.device(device))

# true_deformations = 5*torch.randn((dataset_size,3*N_input_domains))
conformation1 = torch.tensor(np.array([[-8, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), dtype=torch.float32)
# conformation1 = torch.tensor(np.array([[-7, -7, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0]]), dtype=torch.float32)
conformation2 = torch.tensor(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=torch.float32)
conformation1_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
# conformation1_rotation_angle = torch.tensor(np.array([np.pi/4, 0, np.pi/8, 0]), dtype=torch.float32)
conformation1_rotation_angle = torch.tensor(np.array([-np.pi / 4, 0, np.pi / 2, 0]), dtype=torch.float32)
# conformation1_rotation_angle = torch.tensor(np.array([0, 0, 0, 0]))
conformation1_rotation_axis_angle = conformation1_rotation_axis * conformation1_rotation_angle[:, None]
conformation1_rotation_matrix = axis_angle_to_matrix(conformation1_rotation_axis_angle)

conformation2_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
conformation2_rotation_angle = torch.tensor(np.array([0, 0, -np.pi/2, 0]), dtype=torch.float32)
#conformation2_rotation_angle = torch.tensor(np.array([0, 0, 0, 0]))
conformation2_rotation_axis_angle = conformation2_rotation_axis * conformation2_rotation_angle[:, None]
conformation2_rotation_matrix = axis_angle_to_matrix(conformation2_rotation_axis_angle)

conformation1_rotation_matrix = torch.broadcast_to(conformation1_rotation_matrix, (1000, 4, 3, 3))
conformation2_rotation_matrix = torch.broadcast_to(conformation2_rotation_matrix, (1000, 4, 3, 3))
conformation_rotation_matrix = torch.cat([conformation1_rotation_matrix, conformation2_rotation_matrix], dim=0)
conformation1 = torch.broadcast_to(conformation1, (1000, 12))
conformation2 = torch.broadcast_to(conformation2, (1000, 12))
true_deformations = torch.cat([conformation1, conformation2], dim=0)
rotation_angles = torch.tensor(np.random.uniform(0, 2*np.pi, size=(2000,1)), dtype=torch.float32, device=device)
#rotation_angles = torch.tensor(np.random.uniform(0, 2*np.pi, size=(10000)), dtype=torch.float32, device=device)
rotation_axis = torch.randn(size=(2000, 3), device=device)
rotation_axis = rotation_axis/torch.sqrt(torch.sum(rotation_axis**2, dim=1))[:, None]
axis_angle_format = rotation_axis*rotation_angles
rotation_matrices = axis_angle_to_matrix(axis_angle_format)

training_set = true_deformations.to(device)
training_rotations_matrices = rotation_matrices.to(device)
training_rotations_angles = rotation_angles.to(device)
training_rotations_axis = rotation_axis.to(device)
training_conformation_rotation_matrix = conformation_rotation_matrix.to(device)
torch.save(training_set, dataset_path + "test_set.npy")
torch.save(training_rotations_angles, dataset_path + "test_rotations_angles.npy")
torch.save(training_rotations_axis, dataset_path + "test_rotations_axis.npy")
torch.save(training_rotations_matrices, dataset_path + "test_rotations_matrices.npy")
torch.save(training_conformation_rotation_matrix, dataset_path + "test_conformation_rotation_matrices.npy")
# torch.save(test_set, dataset_path + "test_set.npy")



training_indexes = torch.tensor(np.array(range(2000)))
all_losses = []
all_rmsd = []
all_dkl_losses = []
all_tau = []
all_latent_distrib = []
all_indexes = []

all_cluster_means_loss = []
all_cluster_std_loss = []
all_cluster_proportions_loss = []
all_lr = []

relative_positions = torch.matmul(absolute_positions, local_frame)
for epoch in range(0, 1):
    epoch_loss = torch.empty(20)
    # data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    data_loader = iter(DataLoader(training_indexes, batch_size=batch_size, shuffle=False))
    for i in range(20):
        start = time.time()
        print("epoch:", epoch)
        print(i / 20)
        # batch_data = next(iter(data_loader))
        batch_indexes = next(data_loader)
        print(batch_indexes)
        ##Getting the batch translations, rotations and corresponding rotation matrices
        batch_data = training_set[batch_indexes]
        batch_rotations_angles = training_rotations_angles[batch_indexes]
        batch_rotations_axis = training_rotations_axis[batch_indexes]
        batch_rotation_matrices = training_rotations_matrices[batch_indexes]
        batch_data_for_deform = torch.reshape(batch_data, (batch_size, N_input_domains, 3))
        batch_conformation_rotation_matrices = training_conformation_rotation_matrix[batch_indexes]
        ## Deforming the structure for each batch data point
        deformed_structures = utils.deform_structure(absolute_positions, cutoff1, cutoff2, batch_data_for_deform,
                                                     batch_conformation_rotation_matrices, local_frame,
                                                     relative_positions,
                                                     1510, device)

        print("Deformed")
        ## We then rotate the structure and project them on the x-y plane.
        deformed_images = renderer.compute_x_y_values_all_atoms(deformed_structures, batch_rotation_matrices)
        print("images")

        #Getting the latent distrib parameters:
        latent_distrib = network.encode(deformed_images)
        all_latent_distrib.append(latent_distrib.detach().numpy())
        all_indexes.append(batch_indexes.detach().numpy())

        new_structure, mask_weights, translations, latent_distrib_parameters, latent_mean, latent_std \
            = network.forward(batch_indexes, deformed_images)

        loss, rmsd, Dkl_loss, Dkl_mask_mean, Dkl_mask_std, Dkl_mask_proportions = network.loss(
            new_structure, mask_weights, deformed_images, batch_indexes, batch_rotation_matrices, latent_mean,
            latent_std)
        epoch_loss[i] = loss
        # print("Translation network:", translations[k, :, :])
        # print("True translations:", torch.reshape(training_set[ind, :],(batch_size, N_input_domains, 3) )[k,:,:])
        # print("Mask weights:",network.multiply_windows_weights())
        # print("Total loss:",loss)
        print("Printing metrics")
        all_losses.append(loss.cpu().detach())
        all_dkl_losses.append(Dkl_loss.cpu().detach())
        all_rmsd.append(rmsd.cpu().detach())
        all_tau.append(network.tau)
        all_cluster_means_loss.append(Dkl_mask_mean.cpu().detach())
        all_cluster_std_loss.append(Dkl_mask_std.cpu().detach())
        all_cluster_proportions_loss.append(Dkl_mask_proportions.cpu().detach())
        # print("Lat mean:", network.latent_mean)
        # print("Lat std:", network.latent_std)
        end = time.time()
        print("Running time one iteration:", end - start)
        # print(network.weights.requires_grad)
        # network.weights.requires_grad = False
        print("\n\n")
        np.save(dataset_path + "losses_train.npy", np.array(all_losses))
        np.save(dataset_path +"losses_dkl.npy", np.array(all_dkl_losses))
        np.save(dataset_path +"losses_rmsd.npy", np.array(all_rmsd))
        np.save(dataset_path + "losses_cluster_mean", np.array(all_cluster_means_loss))
        np.save(dataset_path + "losses_cluster_std", np.array(all_cluster_std_loss))
        np.save(dataset_path + "losses_cluster_proportions", np.array(all_cluster_proportions_loss))
        np.save(dataset_path +"all_tau.npy", np.array(all_tau))
        np.save(dataset_path + "all_lr.npy", np.array(all_lr))
        #np.save("data/losses_test.npy", np.array(losses_test))
        mask = network.compute_mask()
        mask_python = mask.to("cpu").detach()
        np.save(dataset_path +"mask"+str(epoch)+".npy", mask_python)
        torch.save(network.state_dict(), dataset_path +"model")
        torch.save(network, dataset_path +"full_model")
        np.save(dataset_path + "latent_distrib.npy", np.array(all_latent_distrib))
        np.save(dataset_path + "indexes.npy", np.array(all_indexes))