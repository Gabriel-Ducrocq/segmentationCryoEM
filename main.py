#from MPNN import MessagePassingNetwork
import Bio.PDB.vectors
import matplotlib.pyplot as plt

import utils
from mlp import MLP
from network import Net
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import time
from imageRenderer import Renderer
from pytorch3d.transforms import axis_angle_to_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 250
#This represent the number of true domains
N_domains = 3
N_pixels = 64*64
#This represents the number of domain we think there are
N_input_domains = 4
latent_dim = 9
num_nodes = 1510
cutoff1 = 300
cutoff2 = 1000
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
#B = 10
B = 100
S = 1
dataset_size = 10000
test_set_size = int(dataset_size/10)

print("Is cuda available ?", torch.cuda.is_available())

def train_loop(network, absolute_positions, renderer, local_frame, generate_dataset=True,
               dataset_path="data/imagesGMMRotations100000/"):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=300)
    all_losses = []
    all_rmsd = []
    all_dkl_losses = []
    all_mask_loss = []
    all_tau = []

    relative_positions = torch.matmul(absolute_positions, local_frame)

    if generate_dataset:
        #true_deformations = 5*torch.randn((dataset_size,3*N_input_domains))
        conformation1 = torch.tensor(np.array([[-7, -7, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0]]), dtype=torch.float32)
        conformation2 = torch.tensor(np.array([7, -7, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=torch.float32)
        conformation1_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
        #conformation1_rotation_angle = torch.tensor(np.array([np.pi/4, 0, np.pi/8, 0]), dtype=torch.float32)
        conformation1_rotation_angle = torch.tensor(np.array([np.pi / 4, 0, 0, 0]), dtype=torch.float32)
        #conformation1_rotation_angle = torch.tensor(np.array([0, 0, 0, 0]))
        conformation1_rotation_axis_angle = conformation1_rotation_axis*conformation1_rotation_angle[:, None]
        conformation1_rotation_matrix = axis_angle_to_matrix(conformation1_rotation_axis_angle)
        conformation2_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
        conformation2_rotation_angle = torch.tensor(np.array([-np.pi/4, 0, 0, 0]), dtype=torch.float32)
        #conformation2_rotation_angle = torch.tensor(np.array([0, 0, 0, 0]))
        conformation2_rotation_axis_angle = conformation2_rotation_axis * conformation2_rotation_angle[:, None]
        conformation2_rotation_matrix = axis_angle_to_matrix(conformation2_rotation_axis_angle)

        conformation1_rotation_matrix = torch.broadcast_to(conformation1_rotation_matrix, (12500, 4, 3, 3))
        conformation2_rotation_matrix = torch.broadcast_to(conformation2_rotation_matrix, (12500, 4, 3, 3))
        conformation_rotation_matrix = torch.cat([conformation1_rotation_matrix, conformation2_rotation_matrix], dim=0)
        conformation1 = torch.broadcast_to(conformation1, (12500, 12))
        conformation2 = torch.broadcast_to(conformation2, (12500, 12))
        true_deformations = torch.cat([conformation1, conformation2], dim=0)
        rotation_angles = torch.tensor(np.random.uniform(0, 2*np.pi, size=(25000,1)), dtype=torch.float32, device=device)
        #rotation_angles = torch.tensor(np.random.uniform(0, 2*np.pi, size=(10000)), dtype=torch.float32, device=device)
        rotation_axis = torch.randn(size=(25000, 3), device=device)
        rotation_axis = rotation_axis/torch.sqrt(torch.sum(rotation_axis**2, dim=1))[:, None]
        axis_angle_format = rotation_axis*rotation_angles
        rotation_matrices = axis_angle_to_matrix(axis_angle_format)
        #rotation_matrices = torch.zeros((10000, 3, 3))
        #rotation_matrices[:, 0, 0] = torch.cos(rotation_angles)
        #rotation_matrices[:, 1, 1] = torch.cos(rotation_angles)
        #rotation_matrices[:, 1, 0] = torch.sin(rotation_angles)
        #rotation_matrices[:, 0, 1] = -torch.sin(rotation_angles)
        #rotation_matrices[:, 2, 2] = 1
        #rotation_angles = rotation_angles[:, None]


        training_set = true_deformations.to(device)
        rotation_matrices = torch.transpose(rotation_matrices, dim0=-2, dim1=-1)
        training_rotations_matrices = rotation_matrices.to(device)
        training_rotations_angles = rotation_angles.to(device)
        training_rotations_axis = rotation_axis.to(device)
        conformation_rotation_matrix = torch.transpose(conformation_rotation_matrix, dim0=-2, dim1=-1)
        training_conformation_rotation_matrix = conformation_rotation_matrix.to(device)


        torch.save(training_set, dataset_path + "training_set.npy")
        torch.save(training_rotations_angles, dataset_path + "training_rotations_angles.npy")
        torch.save(training_rotations_axis, dataset_path + "training_rotations_axis.npy")
        torch.save(training_rotations_matrices, dataset_path + "training_rotations_matrices.npy")
        torch.save(training_conformation_rotation_matrix, dataset_path + "training_conformation_rotation_matrices.npy")
        #torch.save(test_set, dataset_path + "test_set.npy")

    training_set = torch.load(dataset_path + "training_set.npy").to(device)
    training_rotations_angles = torch.load(dataset_path + "training_rotations_angles.npy").to(device)
    training_rotations_axis = torch.load(dataset_path + "training_rotations_axis.npy").to(device)
    training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices.npy").to(device)
    training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices.npy")
    print("Creating dataset")
    print("Done creating dataset")
    training_indexes = torch.tensor(np.array(range(25000)))
    for epoch in range(0,5000):
        epoch_loss = torch.empty(100)
        #data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        data_loader = DataLoader(training_indexes, batch_size=batch_size, shuffle=True)
        for i in range(100):
            start = time.time()
            print("epoch:", epoch)
            print(i/100)
            #batch_data = next(iter(data_loader))
            batch_indexes = next(iter(data_loader))
            ##Getting the batch translations, rotations and corresponding rotation matrices
            batch_data = training_set[batch_indexes]
            batch_rotations_angles = training_rotations_angles[batch_indexes]
            batch_rotations_axis = training_rotations_axis[batch_indexes]
            batch_rotation_matrices = training_rotations_matrices[batch_indexes]
            batch_data_for_deform = torch.reshape(batch_data, (batch_size, N_input_domains, 3))
            batch_conformation_rotation_matrices = training_conformation_rotation_matrix[batch_indexes]
            ## Deforming the structure for each batch data point
            deformed_structures = utils.deform_structure(absolute_positions, cutoff1, cutoff2,batch_data_for_deform,
                                                         batch_conformation_rotation_matrices, local_frame, relative_positions,
                                                         1510, device)


            print("Deformed")
            ## We then rotate the structure and project them on the x-y plane.
            deformed_images = renderer.compute_x_y_values_all_atoms(deformed_structures, batch_rotation_matrices)
            #print(batch_rotations[0])
            #print(batch_data)
            #plt.imshow(deformed_images[0], cmap="gray")
            #plt.show()
            print("images")
            #new_structure, mask_weights, translations, latent_distrib_parameters = network.forward(deformed_images)
            new_structure, mask_weights, translations, latent_distrib_parameters = network.forward(batch_indexes,
                                                                                            batch_rotations_angles,
                                                                                            batch_rotations_axis)
            #print("Mask weights")
            #print(mask_weights)
            #b = np.argmax(mask_weights.detach().numpy(), axis=1)
            #print(np.sum(b==0))
            #print(np.sum(b == 1))
            #print(np.sum(b == 2))
            #print(np.sum(b == 3))
            #loss, rmsd, Dkl_loss = network.loss(new_structure, deformed_images, latent_distrib_parameters)
            loss, rmsd, Dkl_loss, mask_loss = network.loss(new_structure, mask_weights,deformed_images, batch_indexes,
                                                           batch_rotation_matrices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k = np.random.randint(0, 6000)
            epoch_loss[i] = loss
            #print("Translation network:", translations[k, :, :])
            #print("True translations:", torch.reshape(training_set[ind, :],(batch_size, N_input_domains, 3) )[k,:,:])
            #print("Mask weights:",network.multiply_windows_weights())
            #print("Total loss:",loss)
            print("Printing metrics")
            all_losses.append(loss.cpu().detach())
            all_dkl_losses.append(Dkl_loss.cpu().detach())
            all_rmsd.append(rmsd.cpu().detach())
            all_mask_loss.append(mask_loss.cpu().detach())
            all_tau.append(network.tau)
            #print("Lat mean:", network.latent_mean)
            #print("Lat std:", network.latent_std)
            end = time.time()
            print("Gradient of mask:")
            print(torch.sum(network.cluster_means.grad)**2)
            print("Running time one iteration:", end-start)
            #print(network.weights.requires_grad)
            #network.weights.requires_grad = False
            print("\n\n")

            #writer.add_scalar('Loss/train', loss, i)
            #writer.add_scalar('Loss/test', np.random.random(), n_iter)
            #writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
            #writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

        scheduler.step(torch.mean(epoch_loss))

        #if (epoch+1)%50 == 0:
        #    network.weights.requires_grad = not network.weights.requires_grad
        #    network.latent_std.requires_grad = not network.latent_std.requires_grad
        #    network.latent_mean.requires_grad = not network.latent_mean.requires_grad

        if (epoch+1)%100 == 0:
            network.tau = network.annealing_tau * network.tau

        #if epoch+1 == 30:
        #    for g in optimizer.param_groups:
        #        g['lr'] = 0.001

        #test_set_normed = (test_set - avg)/std

        #test_set_normed = test_set
        #new_structure, mask_weights, translations = network.forward(nodes_features, edge_indexes, edges_features,
        #                                                            test_set_normed)
        #true_deformation = torch.reshape(test_set, (test_set_normed.shape[0], N_domains, 3))
        #loss_test = network.loss(new_structure, true_deformation, mask_weights, False)
        #losses_test.append(loss_test.to("cpu").detach())
        #print("Loss test:", loss_test)
        print("\n\n\n\n")
        np.save(dataset_path + "losses_train.npy", np.array(all_losses))
        np.save(dataset_path +"losses_dkl.npy", np.array(all_dkl_losses))
        np.save(dataset_path +"losses_rmsd.npy", np.array(all_rmsd))
        np.save(dataset_path +"losses_mask.npy", np.array(all_mask_loss))
        np.save(dataset_path +"all_tau.npy", np.array(all_tau))
        #np.save("data/losses_test.npy", np.array(losses_test))
        mask = network.compute_mask()
        mask_python = mask.to("cpu").detach()
        np.save(dataset_path +"mask"+str(epoch)+".npy", mask_python)
        #scheduler.step(loss_test)
        torch.save(network.state_dict(), dataset_path +"model")
        torch.save(network, dataset_path +"full_model")
        #scheduler.step(loss_test)

def experiment(graph_file="data/features.npy"):
    features = np.load(graph_file, allow_pickle=True)
    features = features.item()
    absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
    absolute_positions = absolute_positions.to(device)
    local_frame = torch.tensor(features["local_frame"])
    local_frame = local_frame.to(device)

    translation_mlp = MLP(latent_dim + 4, 2*3*N_input_domains, 350, device, num_hidden_layers=2)
    encoder_mlp = MLP(N_pixels, latent_dim*2, 1024, device, num_hidden_layers=4)

    pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
    pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
    renderer = Renderer(pixels_x, pixels_y, std=1, device=device)

    net = Net(num_nodes, N_input_domains, latent_dim, B, S, encoder_mlp, translation_mlp, renderer, local_frame,
              absolute_positions, batch_size, cutoff1, cutoff2, device)
    net.to(device)
    train_loop(net, absolute_positions, renderer, local_frame)


if __name__ == '__main__':
    experiment()
