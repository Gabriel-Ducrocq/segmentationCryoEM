#from MPNN import MessagePassingNetwork
from mlp import MLP
from network import Net
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 200
N_domains = 3
latent_dim = 3*N_domains
num_nodes = 1510
cutoff1 = 300
cutoff2 = 1000
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
B = 200
S = 1
dataset_size = 100000
test_set_size = int(dataset_size/10)


def train_loop(network, absolute_positions, nodes_features, edge_indexes, edges_features, latent_variables,
               generate_dataset=True, dataset_path="data/"):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(network.parameters(), lr=5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    all_losses = []
    losses_test = []
    if generate_dataset:
        #latent_vars = 4*torch.randn((dataset_size,3*N_domains))
        latent_vars = torch.zeros((dataset_size,3*N_domains))
        latent_vars.to(device)
        latent_vars[:, :3] = 5
        latent_vars[:, 3:6] = -5
        latent_vars[:, 6:] = 10
        training_set = latent_vars[test_set_size:]
        test_set = latent_vars[:test_set_size]

        torch.save(training_set, dataset_path + "training_set.npy")
        torch.save(test_set, dataset_path + "test_set.npy")

    training_set = torch.load(dataset_path + "training_set.npy")
    test_set = torch.load(dataset_path + "test_set.npy")

    std = torch.std(training_set, dim=0)
    avg = torch.mean(training_set, dim=0)

    for epoch in range(1000):
        trainingDataLoader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        for i in range(500):
            print("epoch:", epoch)
            print(i/500)
            #print(network.multiply_windows_weights())
            latent_vars = next(iter(trainingDataLoader))
            #latent_vars_normed = (latent_vars - avg)/std
            latent_vars_normed = latent_vars
            new_structure, mask_weights, translations = network.forward(nodes_features, edge_indexes, edges_features,
                                                                        latent_vars_normed)
            true_deformation = torch.reshape(latent_vars, (batch_size, N_domains, 3))
            loss = network.loss(new_structure, true_deformation, mask_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            k = np.random.randint(0, 200)
            print(translations[k, :, :])
            print(true_deformation[k, :, :]**3)
            print(network.multiply_windows_weights())
            print(loss)
            all_losses.append(loss.detach())
            print("\n\n")


        #test_set_normed = (test_set - avg)/std
        test_set_normed = test_set
        new_structure, mask_weights, translations = network.forward(nodes_features, edge_indexes, edges_features,
                                                                    test_set_normed)
        true_deformation = torch.reshape(test_set, (test_set_normed.shape[0], N_domains, 3))
        loss_test = network.loss(new_structure, true_deformation, mask_weights, False)
        losses_test.append(loss_test.detach())
        print("Loss test:", loss_test)
        print("\n\n\n\n")
        np.save("data/losses_train.npy", np.array(all_losses))
        np.save("data/losses_test.npy", np.array(losses_test))
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
    absolute_positions.to(device)
    local_frame = torch.tensor(features["local_frame"])

    message_mlp = MLP(30, 50, 100, num_hidden_layers=2)
    update_mlp = MLP(62, 50, 200, num_hidden_layers=2)
    #translation_mlp = MLP(53, 3, 100, num_hidden_layers=2)
    translation_mlp = MLP(9, 9, 350, num_hidden_layers=6)

    #mpnn = MessagePassingNetwork(message_mlp, update_mlp, num_nodes, num_edges, latent_dim = 3)
    net = Net(num_nodes, N_domains, B, S, None, translation_mlp, local_frame, absolute_positions, batch_size, cutoff1, cutoff2)
    net.to(device)
    train_loop(net, absolute_positions, nodes_features, edge_indexes, edges_features, torch.ones((10, 3)))


if __name__ == '__main__':
    experiment()
