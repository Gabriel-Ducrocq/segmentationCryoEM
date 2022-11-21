#from MPNN import MessagePassingNetwork
from mlp import MLP
from network import Net
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision


writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 500
N_domains = 3
latent_dim = 3*N_domains
num_nodes = 1510
cutoff1 = 300
cutoff2 = 1000
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
B = 10
S = 1
dataset_size = 100000
test_set_size = int(dataset_size/10)


def train_loop(network, absolute_positions, nodes_features, edge_indexes, edges_features, latent_variables,
               generate_dataset=True, dataset_path="data/"):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(network.parameters(), lr=5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    all_losses = []
    all_dkl_losses = []
    all_rmsd_losses = []
    all_mask_losses = []
    losses_test = []

    if generate_dataset:
        latent_vars = 16*torch.randn((dataset_size,3*N_domains))
        #latent_vars = torch.empty((dataset_size,3*N_domains))
        #latent_vars[:, :3] = 5
        #latent_vars[:, 3:6] = -5
        #latent_vars[:, 6:] = 10
        latent_vars.to(device)

        training_set = latent_vars.to(device)
        #training_set = latent_vars[test_set_size:]
        #test_set = latent_vars[:test_set_size]

        torch.save(training_set, dataset_path + "training_set.npy")
        #torch.save(test_set, dataset_path + "test_set.npy")

    training_set = torch.load(dataset_path + "training_set.npy").to(device)
    #test_set = torch.load(dataset_path + "test_set.npy").to(device)

    #std = torch.std(training_set, dim=0)
    #avg = torch.mean(training_set, dim=0)
    #indexes = torch.linspace(0, 90000, steps=1, dtype=torch.long)
    indexes = torch.tensor(np.array(range(90000)), device=device)
    for epoch in range(1000):
        epoch_loss = torch.empty(180)
        indexesDataLoader = DataLoader(indexes, batch_size=batch_size, shuffle=True)#, pin_memory=True)
        #print("epoch:", epoch)
        for i in range(180):
            start = time.time()
            print("epoch:", epoch)
            print(i/180)
            #print(network.multiply_windows_weights())
            ind = next(iter(indexesDataLoader))
            #latent_vars_normed = (latent_vars - avg)/std
            latent_vars_normed = network.sample_q(ind)
            new_structure, mask_weights, translations = network.forward(nodes_features, edge_indexes, edges_features,
                                                                        latent_vars_normed)
            #true_deformation = torch.reshape(latent_vars, (batch_size, N_domains, 3))
            loss, rmsd_loss, dkl_loss, mask_loss = network.loss(new_structure, torch.reshape(training_set[ind, :],(batch_size, 3, 3) ), mask_weights, ind)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k = np.random.randint(0, 500)
            epoch_loss[i] = loss
            print(translations[k, :, :])
            #print(true_deformation[k, :, :]**3)
            print(network.multiply_windows_weights())
            print(loss)
            all_losses.append(loss.detach())
            all_dkl_losses.append(dkl_loss.detach())
            all_rmsd_losses.append(rmsd_loss.detach())
            all_mask_losses.append(mask_loss.detach())
            #print(network.latent_std.shape)
            print("Lat mean:", network.latent_mean)
            print("Lat std:", network.latent_std)
            print("\n\n")
            end = time.time()
            print(end-start)

            #writer.add_scalar('Loss/train', loss, i)
            #writer.add_scalar('Loss/test', np.random.random(), n_iter)
            #writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
            #writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

        scheduler.step(torch.mean(epoch_loss))
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
        np.save("data/losses_dkl_train.npy", np.array(all_dkl_losses))
        np.save("data/losses_rmsd_train.npy", np.array(all_rmsd_losses))
        np.save("data/losses_mask_train.npy", np.array(all_mask_losses))
        #np.save("data/losses_test.npy", np.array(losses_test))
        mask = network.multiply_windows_weights()
        mask_python = mask.to("cpu").detach()
        np.save("data/mask"+str(epoch)+".npy", mask_python)
        #scheduler.step(loss_test)
        torch.save(network.state_dict(), "model")
        torch.save(network, "entire_model")
        #scheduler.step(loss_test)

def experiment(graph_file="data/features.npy"):
    features = np.load(graph_file, allow_pickle=True)
    features = features.item()
    nodes_features = torch.tensor(features["nodes_features"], dtype=torch.float)
    edges_features = torch.tensor(features["edges_features"], dtype=torch.float)
    edge_indexes = torch.tensor(features["edge_indexes"], dtype=torch.long)
    absolute_positions = torch.tensor(features["absolute_positions"])
    absolute_positions = absolute_positions.to(device)
    local_frame = torch.tensor(features["local_frame"])
    local_frame = local_frame.to(device)

    #message_mlp = MLP(30, 50, 100, num_hidden_layers=2)
    #update_mlp = MLP(62, 50, 200, num_hidden_layers=2)
    #translation_mlp = MLP(53, 3, 100, num_hidden_layers=2)
    translation_mlp = MLP(9, 9, 350, device, num_hidden_layers=2)


    #mpnn = MessagePassingNetwork(message_mlp, update_mlp, num_nodes, num_edges, latent_dim = 3)
    net = Net(num_nodes, N_domains, B, S, None, translation_mlp, local_frame, absolute_positions, batch_size, cutoff1, cutoff2, device)
    net.to(device)
    train_loop(net, absolute_positions, nodes_features, edge_indexes, edges_features, torch.ones((10, 3)))


if __name__ == '__main__':
    experiment()
