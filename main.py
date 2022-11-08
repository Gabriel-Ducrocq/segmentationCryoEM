from MPNN import MessagePassingNetwork
from mlp import MLP
from network import Net
import numpy as np
import torch
import torch.optim.lr_scheduler

batch_size = 10
N_domains = 2
latent_dim = 3*N_domains
num_nodes = 1510
cutoff = int(num_nodes/4)
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
B = 200
S = 1


def train_loop(network, absolute_positions, nodes_features, edge_indexes, edges_features, latent_variables):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    all_losses = []
    latent_vars = 4*torch.randn((10,3*N_domains))
    std = torch.std(latent_vars, dim= 0)
    avg = torch.mean(latent_vars, dim = 0)
    latent_vars_norm = (latent_vars - avg)/std
    for epoch in range(1000):
        ##Be careful: only a hundred iterations !
        for i in range(1000):
            #k = np.random.randint(0, 10)
            #latent_var = latent_vars[k, :]
            #latent_var_norm = (latent_var - avg)/std
            #latent_var_norm = latent_var
            print("epoch:", epoch)
            print(i/1000)
            print(network.multiply_windows_weights())
            new_structure, mask_weights, translations = network.forward(nodes_features, edge_indexes, edges_features,
                                                                        latent_vars_norm)
            true_deformation = torch.reshape(latent_vars, (batch_size, N_domains, 3))
            loss = network.loss(new_structure, true_deformation, mask_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            k = np.random.randint(0, batch_size)
            print("Net translations:", translations[k, :, :])
            print("True translations:", true_deformation[k, :, :])
            print(loss)
            print("\n\n")

        mask = network.multiply_windows_weights()
        mask_python = mask.detach()
        np.save("data/mask"+str(epoch)+".npy", mask_python)
        scheduler.step()

def experiment(graph_file="data/features.npy"):
    features = np.load(graph_file, allow_pickle=True)
    features = features.item()
    nodes_features = torch.tensor(features["nodes_features"], dtype=torch.float)
    edges_features = torch.tensor(features["edges_features"], dtype=torch.float)
    edge_indexes = torch.tensor(features["edge_indexes"], dtype=torch.long)
    absolute_positions = torch.tensor(features["absolute_positions"])
    local_frame = torch.tensor(features["local_frame"])

    message_mlp = MLP(30, 50, 100, num_hidden_layers=2)
    update_mlp = MLP(62, 50, 200, num_hidden_layers=2)
    #translation_mlp = MLP(53, 3, 100, num_hidden_layers=2)
    translation_mlp = MLP(6, 6, 100, num_hidden_layers=2)

    mpnn = MessagePassingNetwork(message_mlp, update_mlp, num_nodes, num_edges, latent_dim = 3)
    net = Net(num_nodes, N_domains, B, S, mpnn, translation_mlp, local_frame, absolute_positions, batch_size, cutoff)
    train_loop(net, absolute_positions, nodes_features, edge_indexes, edges_features, torch.ones((10, 3)))


if __name__ == '__main__':
    experiment()
