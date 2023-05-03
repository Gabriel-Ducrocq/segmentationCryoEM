import numpy as np
import torch


relative_positions = torch.matmul(absolute_positions, local_frame)
generated_noise = torch.randn(size=(10000, 64, 64)) * np.sqrt(0.2)

conformation1 = torch.tensor(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), dtype=torch.float32)
conformation2 = torch.tensor(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=torch.float32)
conformation1_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
                                           dtype=torch.float32)
# conformation1_rotation_angle = torch.tensor(np.array([-np.pi / 4, 0, np.pi/2, 0]), dtype=torch.float32)
conformation1_rotation_angle = torch.zeros((5000, 4), dtype=torch.float32)
conformation1_rotation_angle[:, 2] = -torch.rand(size=(5000,)) * torch.pi
# conformation1_rotation_axis_angle = conformation1_rotation_axis*conformation1_rotation_angle[:, None]
conformation1_rotation_axis_angle = torch.broadcast_to(conformation1_rotation_axis[None, :, :], (5000, 4, 3)) \
                                    * conformation1_rotation_angle[:, :, None]

conformation1_rotation_matrix = axis_angle_to_matrix(conformation1_rotation_axis_angle)

# conformation2_rotation_axis = torch.tensor(np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]), dtype=torch.float32)
# conformation2_rotation_angle = torch.tensor(np.array([0, 0, -np.pi/2, 0]), dtype=torch.float32)
conformation2_rotation_angle = torch.zeros((5000, 4), dtype=torch.float32)
conformation2_rotation_angle[:, 2] = -torch.rand(size=(5000,)) * torch.pi
# conformation2_rotation_axis_angle = conformation2_rotation_axis * conformation2_rotation_angle[:, None]
conformation2_rotation_axis_angle = torch.broadcast_to(conformation1_rotation_axis[None, :, :], (5000, 4, 3)) \
                                    * conformation1_rotation_angle[:, :, None]
conformation2_rotation_matrix = axis_angle_to_matrix(conformation2_rotation_axis_angle)

# conformation1_rotation_matrix = torch.broadcast_to(conformation1_rotation_matrix, (5000, 4, 3, 3))
# conformation2_rotation_matrix = torch.broadcast_to(conformation2_rotation_matrix, (5000, 4, 3, 3))
conformation_rotation_matrix = torch.cat([conformation1_rotation_matrix, conformation2_rotation_matrix], dim=0)
conformation1 = torch.broadcast_to(conformation1, (5000, 12))
conformation2 = torch.broadcast_to(conformation2, (5000, 12))
true_deformations = torch.cat([conformation1, conformation2], dim=0)
rotation_angles = torch.tensor(np.random.uniform(0, 2 * np.pi, size=(10000, 1)), dtype=torch.float32, device=device)
# rotation_angles = torch.tensor(np.random.uniform(0, 2*np.pi, size=(10000)), dtype=torch.float32, device=device)
rotation_axis = torch.randn(size=(10000, 3), device=device)
rotation_axis = rotation_axis / torch.sqrt(torch.sum(rotation_axis ** 2, dim=1))[:, None]
axis_angle_format = rotation_axis * rotation_angles
rotation_matrices = axis_angle_to_matrix(axis_angle_format)
# rotation_matrices = torch.zeros((10000, 3, 3))
# rotation_matrices[:, 0, 0] = torch.cos(rotation_angles)
# rotation_matrices[:, 1, 1] = torch.cos(rotation_angles)
# rotation_matrices[:, 1, 0] = torch.sin(rotation_angles)
# rotation_matrices[:, 0, 1] = -torch.sin(rotation_angles)
# rotation_matrices[:, 2, 2] = 1
# rotation_angles = rotation_angles[:, None]

training_set = true_deformations.to(device)
# rotation_matrices = torch.transpose(rotation_matrices, dim0=-2, dim1=-1)
training_rotations_matrices = rotation_matrices.to(device)
training_rotations_angles = rotation_angles.to(device)
training_rotations_axis = rotation_axis.to(device)
# conformation_rotation_matrix = torch.transpose(conformation_rotation_matrix, dim0=-2, dim1=-1)
training_conformation_rotation_matrix = conformation_rotation_matrix.to(device)

torch.save(training_set, dataset_path + "training_set.npy")
torch.save(training_rotations_angles, dataset_path + "training_rotations_angles.npy")
torch.save(training_rotations_axis, dataset_path + "training_rotations_axis.npy")
torch.save(training_rotations_matrices, dataset_path + "training_rotations_matrices.npy")
torch.save(training_conformation_rotation_matrix, dataset_path + "training_conformation_rotation_matrices.npy")
# torch.save(test_set, dataset_path + "test_set.npy")

training_set = torch.load(dataset_path + "training_set.npy").to(device)
training_rotations_angles = torch.load(dataset_path + "training_rotations_angles.npy").to(device)
training_rotations_axis = torch.load(dataset_path + "training_rotations_axis.npy").to(device)
training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices.npy").to(device)
training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices.npy")
print("Creating dataset")
print("Done creating dataset")
training_indexes = torch.tensor(np.array(range(10000)))
with autograd.detect_anomaly():
    for epoch in range(0, 5000):
        # if epoch == 0:
        #    weight_histograms(writer, epoch, network)
        # else:
        #    weight_histograms(writer, epoch, network, True)

        epoch_loss = torch.zeros(10, device=device)
        # data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        data_loader = iter(DataLoader(training_indexes, batch_size=batch_size, shuffle=True))
        for i in range(100):
            start = time.time()
            print("epoch:", epoch)
            print(i / 100)
            # if epoch == 0:
            #    weight_histograms(writer, epoch, network)
            # else:
            #    weight_histograms(writer, epoch, network, True)

            # batch_data = next(iter(data_loader))
            batch_indexes = next(data_loader)
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
            print("Deformed mean", torch.mean(deformed_images))
            # print(batch_rotations[0])
            # print(batch_data)
            # plt.imshow(deformed_images[0], cmap="gray")
            # plt.show()
            print("images")
            noise_components = generated_noise[batch_indexes].to(device)