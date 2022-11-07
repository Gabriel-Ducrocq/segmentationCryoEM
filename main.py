from MPNN import MessagePassingNetwork
from mlp import MLP
from network import Net
import numpy as np
import torch
import torch.optim.lr_scheduler

N_domains = 2
latent_dim = 3*N_domains
num_nodes = 1510
cutoff = int(num_nodes/2)
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
B = 200
S = 100


def train_loop(network, absolute_positions, nodes_features, edge_indexes, edges_features, latent_variables):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    all_losses = []
    latent_vars = 4*torch.randn((10,3*N_domains)) #np.random.normal(scale=4, size=(1, 3), dtype=float)
    std = torch.std(latent_vars, dim= 0)
    avg = torch.mean(latent_vars, dim = 0)
    for epoch in range(1000):
        for i in range(1000):
            k = np.random.randint(0, 10)
            latent_var = latent_vars[k, :]
            latent_var_norm = (latent_var - avg)/std
            print("epoch:", epoch)
            print(i/1000)
            new_structure, mask_weights, translations = network.forward(nodes_features, edge_indexes, edges_features, latent_var_norm)
            true_deformed_structure = torch.empty_like(absolute_positions)
            true_deformed_structure[:3*cutoff, :] = absolute_positions[:3*cutoff, :] + 5#+ latent_var_norm[:3]**2
            true_deformed_structure[3 * cutoff:, :] = absolute_positions[3 * cutoff:, :] - 5 #+ latent_var_norm[3:]**2
            loss = network.loss(new_structure, true_deformed_structure, mask_weights)
            #loss = network.loss(translations, latent_var ** 2, mask_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(network.weights)
            #for param in network.parameters():
            #    print(param.grad)
            #print(network.decoder.message_mlp.output_layer.weight.grad)
            #print(list(network.parameters()))
            print(network.multiply_windows_weights())
            print("Output:", translations)
            print("Latent var squred:", latent_var**2)
            print("Mean:", torch.mean(latent_vars**2, dim=0))
            print(loss)
            print("\n\n")

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
    translation_mlp = MLP(53, 3, 100, num_hidden_layers=2)

    mpnn = MessagePassingNetwork(message_mlp, update_mlp, num_nodes, num_edges, latent_dim = 3)
    net = Net(num_nodes, N_domains, B, S, mpnn, translation_mlp, local_frame, absolute_positions)
    train_loop(net, absolute_positions, nodes_features, edge_indexes, edges_features, torch.ones((10, 3)))


if __name__ == '__main__':
    experiment()
