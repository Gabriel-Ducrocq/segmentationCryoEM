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
NUM_ACCUMULATION_STEP = 1

batch_size = 100
#This represent the number of true domains
N_domains = 6
N_pixels = 140*140
#This represents the number of domain we think there are
N_input_domains = 6
latent_dim_x = 40
##Latent of w, see DEEP UNSUPERVISED CLUSTERING WITH GAUSSIAN MIXTURE VARIATIONAL AUTOENCODERS
latent_dim_w = 40
N_mixture_components = 4
num_nodes = 1006
dataset_size = 10000
one_latent_per_domain = False
test_set_size = int(dataset_size/10)

print("Is cuda available ?", torch.cuda.is_available())

def train_loop(network, absolute_positions, renderer, local_frame, generate_dataset=True,
               dataset_path="../VAEProtein/data/vaeTwoClustersMDLatent40GMM/"):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    #optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=300)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1, eta_min=0.00003)
    all_losses = []
    all_rmsd = []
    all_dkl_losses = []
    all_tau = []

    all_cluster_means_loss = []
    all_cluster_std_loss = []
    all_cluster_proportions_loss = []
    all_lr = []

    training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices").to(device)
    training_images = torch.load(dataset_path + "continuousConformationDataSet")
    print("TRAINING IMAGES SHAPE", training_images.shape)
    training_indexes = torch.tensor(np.array(range(10000)))
    with torch.autograd.detect_anomaly(check_nan=False):
        for epoch in range(0,5000):
            epoch_loss = torch.empty(1000)
            #data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
            data_loader = iter(DataLoader(training_indexes, batch_size=batch_size, shuffle=True))
            #for i in range(100):
            for idx, batch_indexes in enumerate(data_loader):
                start = time.time()
                print("epoch:", epoch)
                print(idx/100)
                #batch_indexes = next(data_loader)
                deformed_images = training_images[batch_indexes]
                batch_rotation_matrices = training_rotations_matrices[batch_indexes]
                deformed_images = deformed_images.to(device)
                batch_indexes = batch_indexes.to(device)
                #print(batch_rotations[0])
                #print(batch_data)
                #plt.imshow(deformed_images[0], cmap="gray")
                #plt.show()
                print("images")
                #new_structure, mask_weights, translations, latent_distrib_parameters = network.forward(deformed_images)
                new_structure, mask_weights, translations, latent_variables_w, latent_mean_w, latent_std_w, latent_variables_x, latent_mean_x, latent_std_x\
                    = network.forward(batch_indexes, deformed_images)

                loss, rmsd, Dkl_loss, Dkl_mask_mean, Dkl_mask_std, Dkl_mask_proportions = network.loss(
                    new_structure, mask_weights,deformed_images, batch_indexes, batch_rotation_matrices, latent_mean_w, latent_std_w, latent_mean_x, latent_std_x)
                loss = loss/NUM_ACCUMULATION_STEP
                loss.backward()
                #optimizer.step()
                if ((idx + 1) % NUM_ACCUMULATION_STEP == 0) or (idx + 1 == len(data_loader)):
                    # Update Optimizer
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss[idx] = loss.cpu().detach()
                #print("Translation network:", translations[k, :, :])
                #print("True translations:", torch.reshape(training_set[ind, :],(batch_size, N_input_domains, 3) )[k,:,:])
                #print("Mask weights:",network.multiply_windows_weights())
                #print("Total loss:",loss)
                print("Printing metrics")
                all_losses.append(loss.cpu().detach())
                all_dkl_losses.append(Dkl_loss.cpu().detach())
                all_rmsd.append(rmsd.cpu().detach())
                all_tau.append(network.tau)
                all_cluster_means_loss.append(Dkl_mask_mean.cpu().detach())
                all_cluster_std_loss.append(Dkl_mask_std.cpu().detach())
                all_cluster_proportions_loss.append(Dkl_mask_proportions.cpu().detach())
                #print("Lat mean:", network.latent_mean)
                #print("Lat std:", network.latent_std)
                end = time.time()
                print("Running time one iteration:", end-start)
                #print(network.weights.requires_grad)
                #network.weights.requires_grad = False
                print("\n\n")

                #writer.add_scalar('Loss/train', loss, i)
                #writer.add_scalar('Loss/test', np.random.random(), n_iter)
                #writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
                #writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

            #scheduler.step(torch.mean(epoch_loss))
            #scheduler.step()

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
            np.save(dataset_path + "losses_cluster_mean", np.array(all_cluster_means_loss))
            np.save(dataset_path + "losses_cluster_std", np.array(all_cluster_std_loss))
            np.save(dataset_path + "losses_cluster_proportions", np.array(all_cluster_proportions_loss))
            np.save(dataset_path +"all_tau.npy", np.array(all_tau))
            np.save(dataset_path + "all_lr.npy", np.array(all_lr))
            #np.save("data/losses_test.npy", np.array(losses_test))
            mask = network.compute_mask()
            mask_python = mask.to("cpu").detach()
            np.save(dataset_path +"mask"+str(epoch)+".npy", mask_python)
            #scheduler.step(loss_test)
            torch.save(network.state_dict(), dataset_path +"model")
            torch.save(network, dataset_path +"full_model" + str(epoch))
            #scheduler.step(loss_test)

def experiment(graph_file="data/features.npy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = np.load(graph_file, allow_pickle=True)
    features = features.item()
    absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
    absolute_positions = absolute_positions.to(device)
    local_frame = torch.tensor(features["local_frame"])
    local_frame = local_frame.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if one_latent_per_domain:
        translation_mlp = MLP(latent_dim_x, 2 * 3 , 350, device, num_hidden_layers=2,
                              network_type="decoder")
        encoder_mlp = MLP(N_pixels, latent_dim_x * 2*N_input_domains, [2048, 1024, 512, 512], device, num_hidden_layers=4,
                          network_type="encoder")
    else:
        translation_mlp = MLP(latent_dim_x, 2 * 3 * N_input_domains, 350, device, num_hidden_layers=2,
                              network_type="decoder")
        encoder_mlp = MLP(N_pixels, latent_dim_x*2, [2048, 1024, 512, 512], device, num_hidden_layers=4, network_type="encoder")
        encoder_x = MLP(N_pixels, latent_dim_w*2, [2048, 1024, 512, 512], device, num_hidden_layers=4, network_type="encoder")
        nets_x_given_w = [MLP(latent_dim_w, latent_dim_x*2, [latent_dim_w, latent_dim_w, latent_dim_w, latent_dim_w], device, num_hidden_layers=4, network_type="encoder")
                            for _ in range(N_mixture_components)]



    #encoder_mlp = MLP(N_pixels, latent_dim * 2, [57600, 2048, 1024, 512, 512], device, num_hidden_layers=4)

    #pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
    #pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
    pixels_x = np.linspace(-70, 70, num=140).reshape(1, -1)
    pixels_y = np.linspace(-70, 70, num=140).reshape(1, -1)
    renderer = Renderer(pixels_x, pixels_y, std=1, device=device, use_ctf=True, N_heavy=3*1006)

    #net = Net(num_nodes, N_input_domains, latent_dim, encoder_mlp, translation_mlp, renderer, local_frame,
    #          absolute_positions, batch_size, device, use_encoder=False)

    net = Net(num_nodes, N_input_domains, latent_dim_x, encoder_mlp, translation_mlp, renderer, local_frame,
              absolute_positions, batch_size, device, use_encoder=True, one_latent_per_domain=one_latent_per_domain,
              encoder_x = encoder_x, nets_x_given_w=nets_x_given_w, latent_clustering=True,
              n_components_mixture= N_mixture_components, latent_dim_w=latent_dim_w)
    net.to(device)
    train_loop(net, absolute_positions, renderer, local_frame)


if __name__ == '__main__':
    print("Is cuda available ?", torch.cuda.is_available())
    experiment("../VAEProtein/data/vaeContinuousMD/features.npy")
