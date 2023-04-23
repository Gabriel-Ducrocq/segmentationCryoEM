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
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import autograd

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10
#This represent the number of true domains
N_domains = 3
N_pixels = 64*64
#This represents the number of domain we think there are
N_input_domains = 4
latent_dim = 1
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
NUM_ACCUMULATION_STEP = 1

print("Is cuda available ?", torch.cuda.is_available())

def weight_histograms_linear(writer, step, weights, layer_number):
  flattened_weights = weights.flatten()
  tag = f"layer_{layer_number}"
  #writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')
  print(tag, torch.sum(flattened_weights**2))

def mask_histogram(writer, step, model, get_grad=True):
    for typ, parameters in model.cluster_parameters.items():
        avg = parameters["mean"]
        std = parameters["std"]
        for i in range(N_input_domains):
            tag_avg = f"layer_{typ}_mean_{i}"
            tag_std = f"layer_{typ}_std_{i}"
            writer.add_scalar(tag_avg, avg[0][i], global_step=step)
            writer.add_scalar(tag_std, std[0][i], global_step=step)
            if get_grad:
                tag_avg = f"gradient_{typ}_mean_{i}"
                tag_std = f"gradient_{typ}_std_{i}"
                writer.add_scalar(tag_avg, avg.grad[0][i], global_step=step)
                writer.add_scalar(tag_std, std.grad[0][i], global_step=step)

def grad_histograms_linear(writer, step, weights, layer_number):
  print("GRADGRADGRADGRAD")
  flattened_weights = weights.grad.flatten()
  tag = f"grad_{layer_number }"
  #writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')
  print(tag, torch.sum(flattened_weights**2))



def weight_mlp_histogram(writer, step, model, name, get_grad = False):
    weights = model.input_layer[0].weight
    weight_histograms_linear(writer, step, weights, name + "_input_layer")
    for layer_number in range(len(model.linear_relu_stack)):
        layer = model.linear_relu_stack[layer_number][0]
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            weight_histograms_linear(writer, step, weights, name + "_" + str(layer_number))

    weights = model.output_layer[0].weight
    weight_histograms_linear(writer, step, weights, name + "_final")

    if get_grad:
        weights = model.input_layer[0].weight
        grad_histograms_linear(writer, step, weights, name + "_input_layer")
        for layer_number in range(len(model.linear_relu_stack)):
            layer = model.linear_relu_stack[layer_number][0]
            if isinstance(layer, nn.Linear):
                weights = layer.weight
                grad_histograms_linear(writer, step, weights, name + "_" + str(layer_number))

        weights = model.output_layer[0].weight
        grad_histograms_linear(writer, step, weights, name + "_final")




def weight_histograms(writer, step, model, get_grad=False):
    print("Visualizing model weights...")
    # Iterate over all model layers
    weight_mlp_histogram(writer, step, model.encoder, "encoder", get_grad)
    weight_mlp_histogram(writer, step, model.decoder, "decoder", get_grad)
    #mask_histogram(writer, step, model, get_grad=get_grad)

def train_loop(network, dataset_path="data/vaeContinuousNoisyZhongStyle3/"):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=300)
    all_losses = []
    all_rmsd = []
    all_dkl_losses = []
    all_tau = []

    all_cluster_means_loss = []
    all_cluster_std_loss = []
    all_cluster_proportions_loss = []
    all_lr = []

    training_rotations_matrices = torch.load(dataset_path + "rotationPoseDataSet").to(device)
    training_images = torch.load(dataset_path + "continuousConformationDataSet")
    print("SHAPE IMAGES:", training_images.shape)
    print("SHAPE pose rotations:", training_rotations_matrices.shape)
    training_indexes = torch.tensor(np.array(range(10000)))
    restart = False
    if restart:
        network = torch.load(dataset_path + "full_model2310")

    with autograd.detect_anomaly():
        for epoch in range(0,10000):
            if epoch == 0:
                weight_histograms(writer, epoch, network)
            else:
                weight_histograms(writer, epoch, network, True)

            epoch_loss = torch.zeros(100, device=device)
            #data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
            data_loader = iter(DataLoader(training_indexes, batch_size=batch_size, shuffle=True))
            for i in range(1000):
                start = time.time()
                print("epoch:", epoch)
                print(i/1000)
                #batch_data = next(iter(data_loader))
                batch_indexes = next(data_loader)
                ##Getting the batch translations, rotations and corresponding rotation matrices
                batch_rotation_matrices = training_rotations_matrices[batch_indexes]
                deformed_images = training_images[batch_indexes].to(device)

                new_structure, mask_weights, translations, latent_distrib_parameters, latent_mean, latent_std\
                    = network.forward(batch_indexes, deformed_images)

                loss, rmsd, Dkl_loss, Dkl_mask_mean, Dkl_mask_std, Dkl_mask_proportions = network.loss(
                    new_structure, mask_weights,deformed_images, batch_indexes, batch_rotation_matrices, latent_mean, latent_std)

                loss = loss/NUM_ACCUMULATION_STEP
                loss.backward()
                if (i+1)%NUM_ACCUMULATION_STEP == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss[i//NUM_ACCUMULATION_STEP] += loss

                print("Printing metrics")
                all_losses.append(loss.cpu().detach())
                all_dkl_losses.append(Dkl_loss.cpu().detach())
                all_rmsd.append(rmsd.cpu().detach())
                all_tau.append(network.tau)
                all_cluster_means_loss.append(Dkl_mask_mean.cpu().detach())
                all_cluster_std_loss.append(Dkl_mask_std.cpu().detach())
                all_cluster_proportions_loss.append(Dkl_mask_proportions.cpu().detach())
                end = time.time()
                print("Running time one iteration:", end-start)
                print("\n\n")


            #scheduler.step(torch.mean(epoch_loss))
            #scheduler.step()


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
            torch.save(network, dataset_path +"full_model"+str(epoch))
            #scheduler.step(loss_test)

def experiment(graph_file="data/features.npy"):
    features = np.load(graph_file, allow_pickle=True)
    features = features.item()
    absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
    absolute_positions = absolute_positions.to(device)
    local_frame = torch.tensor(features["local_frame"])
    local_frame = local_frame.to(device)

    translation_mlp = MLP(latent_dim, 2*3*N_input_domains, 350, device, num_hidden_layers=2)
    encoder_mlp = MLP(N_pixels, latent_dim*2, [2048, 1024, 512, 512], device, num_hidden_layers=4)
    #encoder_mlp = MLP(N_pixels, latent_dim * 2, [1024, 512, 128], device, num_hidden_layers=4)

    pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
    pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
    renderer = Renderer(pixels_x, pixels_y, std=1, device=device)

    net = Net(num_nodes, N_input_domains, latent_dim, B, S, encoder_mlp, translation_mlp, renderer, local_frame,
              absolute_positions, batch_size, cutoff1, cutoff2, device)
    net.to(device)
    train_loop(net)


if __name__ == '__main__':
    print("Is cuda available ?", torch.cuda.is_available())
    experiment()
