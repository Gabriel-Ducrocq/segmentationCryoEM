#from MPNN import MessagePassingNetwork
import utils
from mlp import MLP
from network import Net
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from imageRenderer import Renderer
import torchvision


writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
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
B = 10
S = 1
dataset_size = 10000
test_set_size = int(dataset_size/10)


def train_loop(network, absolute_positions, renderer, generate_dataset=True, dataset_path="data/"):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)
    all_losses = []
    all_rmsd = []
    all_dkl_losses = []
    all_mask_loss = []
    all_tau = []

    if generate_dataset:
        true_deformations = 5*torch.randn((dataset_size,3*N_input_domains))
        true_deformations[:, 2] = 0
        #true_deformations[:, 0:2] = 5
        true_deformations[:, 5] = 0
        #true_deformations[:, 3:5] = -5
        true_deformations[:, 8] = 0
        #true_deformations[:, 6:8] = 0
        #latent_vars[:33000] += 2
        #latent_vars[33000:66000] -= 2
        #latent_vars[66000:] += np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        #latent_vars = torch.empty((dataset_size,3*N_domains))
        #latent_vars[:, :3] = 5
        #latent_vars[:, 3:6] = -5
        #latent_vars[:, 6:] = 10

        training_set = true_deformations.to(device)


        torch.save(training_set, dataset_path + "training_set.npy")
        #torch.save(test_set, dataset_path + "test_set.npy")

    training_set = torch.load(dataset_path + "training_set.npy").to(device)
    training_indexes = torch.tensor(np.array(range(10000)))
    for epoch in range(0,1000):
        epoch_loss = torch.empty(100)
        #data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        data_loader = DataLoader(training_indexes, batch_size=batch_size, shuffle=True)
        for i in range(100):
            start = time.time()
            print("epoch:", epoch)
            print(i/100)
            #batch_data = next(iter(data_loader))
            batch_indexes = next(iter(data_loader))
            batch_data = training_set[batch_indexes]
            batch_data_for_deform = torch.reshape(batch_data, (batch_size, N_input_domains, 3))
            deformed_structures = utils.deform_structure(absolute_positions, cutoff1, cutoff2,batch_data_for_deform,
                                                         1510, device)

            print("Deformed")
            deformed_images = renderer.compute_x_y_values_all_atoms(deformed_structures)
            print("images")
            #new_structure, mask_weights, translations, latent_distrib_parameters = network.forward(deformed_images)
            new_structure, mask_weights, translations, latent_distrib_parameters = network.forward(batch_indexes)
            #loss, rmsd, Dkl_loss = network.loss(new_structure, deformed_images, latent_distrib_parameters)
            loss, rmsd, Dkl_loss = network.loss(new_structure, deformed_images, batch_indexes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k = np.random.randint(0, 6000)
            epoch_loss[i] = loss
            #print("Translation network:", translations[k, :, :])
            #print("True translations:", torch.reshape(training_set[ind, :],(batch_size, N_input_domains, 3) )[k,:,:])
            #print("Mask weights:",network.multiply_windows_weights())
            #print("Total loss:",loss)
            all_losses.append(loss.cpu().detach())
            all_dkl_losses.append(Dkl_loss.cpu().detach())
            all_rmsd.append(rmsd.cpu().detach())
            #all_mask_loss.append(mask_loss.cpu().detach())
            all_tau.append(network.tau)
            #print("Lat mean:", network.latent_mean)
            #print("Lat std:", network.latent_std)
            end = time.time()
            print("Running time one iteration:", end -start)
            print("\n\n")

            #writer.add_scalar('Loss/train', loss, i)
            #writer.add_scalar('Loss/test', np.random.random(), n_iter)
            #writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
            #writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

        scheduler.step(torch.mean(epoch_loss))
        if (epoch+1)%15 == 0:
            network.tau = network.annealing_tau * network.tau

        #test_set_normed = (test_set - avg)/std

        #test_set_normed = test_set
        #new_structure, mask_weights, translations = network.forward(nodes_features, edge_indexes, edges_features,
        #                                                            test_set_normed)
        #true_deformation = torch.reshape(test_set, (test_set_normed.shape[0], N_domains, 3))
        #loss_test = network.loss(new_structure, true_deformation, mask_weights, False)
        #losses_test.append(loss_test.to("cpu").detach())
        #print("Loss test:", loss_test)
        print("\n\n\n\n")
        np.save("data/losses_train.npy", np.array(all_losses))
        np.save("data/losses_dkl.npy", np.array(all_dkl_losses))
        np.save("data/losses_rmsd.npy", np.array(all_rmsd))
        np.save("data/losses_mask.npy", np.array(all_mask_loss))
        np.save("data/all_tau.npy", np.array(all_tau))
        #np.save("data/losses_test.npy", np.array(losses_test))
        mask = network.multiply_windows_weights()
        mask_python = mask.to("cpu").detach()
        np.save("data/mask"+str(epoch)+".npy", mask_python)
        #scheduler.step(loss_test)
        torch.save(network.state_dict(), "model")
        torch.save(network, "full_model")
        #scheduler.step(loss_test)

def experiment(graph_file="data/features.npy"):
    features = np.load(graph_file, allow_pickle=True)
    features = features.item()
    absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
    absolute_positions = absolute_positions.to(device)
    local_frame = torch.tensor(features["local_frame"])
    local_frame = local_frame.to(device)

    translation_mlp = MLP(latent_dim, 3*N_input_domains, 350, device, num_hidden_layers=2)
    encoder_mlp = MLP(N_pixels, latent_dim*2, 1024, device, num_hidden_layers=4)

    pixels_x = np.linspace(-70, 70, num=64).reshape(1, -1)
    pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
    renderer = Renderer(pixels_x, pixels_y, std=1, device=device)

    net = Net(num_nodes, N_input_domains, latent_dim, B, S, encoder_mlp, translation_mlp, renderer, local_frame,
              absolute_positions, batch_size, cutoff1, cutoff2, device)
    net.to(device)
    train_loop(net, absolute_positions, renderer)


if __name__ == '__main__':
    experiment()
