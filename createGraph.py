import utils
import numpy as np
from collections import OrderedDict
from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.transform import Rotation

K_neighbours = 30
#rbf_sigma = np.linspace(0, 20, 17)[1:]
rbf_sigma = np.array([1, 10, 20])
SLICE_X = slice(0, 3)
SLICE_Y = slice(3, 6)
SLICE_Z = slice(6, 9)


file = "data/ranked_0_round1.pdb"
out_file = "data/features.npy"
parser = PDBParser(PERMISSIVE=0)

structure = parser.get_structure("A", file)

def get_positions(residue, name):
    x = residue["CA"].get_coord()
    y = residue["N"].get_coord()
    if name == "GLY":
        z = residue["C"].get_coord()
        return x,y,z

    z = residue["C"].get_coord()
    return x.copy(),y.copy(),z.copy()


def get_node_features(structure):
    N_residue = 0
    residues_indexes = []
    c_alpha_positions = []
    nodes_features = []

    residue = list(structure.get_residues())[0]
    u, v, w = get_positions(residue, residue.get_resname())
    local_frame = utils.get_orthonormal_basis(v - u, w - u)
    position_local_frame = u
    absolute_positions = []

    for model in structure:
        for chain in model:
            for residue in chain:
                print(N_residue)
                residues_indexes.append(N_residue)
                name = residue.get_resname()
                x, y, z = get_positions(residue, name)
                absolute_positions.append(x)
                absolute_positions.append(y)
                absolute_positions.append(z)

                c_alpha_positions.append(x)
                x -= position_local_frame
                y -= position_local_frame
                z -= position_local_frame
                loc_x = np.dot(x, local_frame)
                loc_y = np.dot(y, local_frame)
                loc_z = np.dot(z, local_frame)
                loc_pos = np.empty(9)
                loc_pos[:3] = loc_x
                loc_pos[3:6] = loc_y
                loc_pos[6:] = loc_z
                nodes_features.append(loc_pos)
                N_residue +=1

    return np.array(nodes_features), np.array(residues_indexes), np.array(c_alpha_positions), local_frame, \
           position_local_frame, np.array(absolute_positions)



def get_edge_features(residues_indexes, pairwise_distances, nodes_features):
    all_nodes_edges_features = []
    all_nodes_neighb = []
    for n_residue in residues_indexes:
        one_node_neighb = []
        print(n_residue/len(residues_indexes))
        sorted_residues = sorted(zip(pairwise_distances[n_residue, :], residues_indexes))
        neighbours_distances, neighbours_indexes = zip(*sorted_residues)
        for neighb, distance in zip(neighbours_indexes[:K_neighbours], neighbours_distances[:K_neighbours]):
            distance_feature_x_x = np.array([utils.gaussian_rbf(nodes_features[n_residue][SLICE_X],
                                                            nodes_features[neighb][SLICE_X],
                                                            sig) for sig in rbf_sigma])
            distance_feature_y_y = np.array([utils.gaussian_rbf(nodes_features[n_residue][SLICE_Y],
                                                            nodes_features[neighb][SLICE_Y],
                                                            sig) for sig in rbf_sigma])

            distance_feature_z_z = np.array([utils.gaussian_rbf(nodes_features[n_residue][SLICE_Z],
                                                            nodes_features[neighb][SLICE_Z],
                                                            sig) for sig in rbf_sigma])

            one_node_neighb.append(neighb)
            edge_features = np.empty(9)
            edge_features[:3] = distance_feature_x_x
            edge_features[3:6] = distance_feature_y_y
            edge_features[6:] = distance_feature_z_z
            all_nodes_edges_features.append(edge_features)

        all_nodes_neighb.append(one_node_neighb)

    return np.array(all_nodes_edges_features), np.array(all_nodes_neighb)


def get_edge_index(nodes_neighb, edges_features):
    edge_index = []
    for i, neighb in enumerate(nodes_neighb):
        source_nodes = neighb
        target_node = np.ones(len(source_nodes), dtype=int) * i
        if len(edge_index) == 0:
            edge_index = np.vstack((source_nodes, target_node))
            continue

        edge_index_add = np.vstack((source_nodes, target_node))
        edge_index = np.hstack((edge_index, edge_index_add))

    return edge_index

#def get_data_x(residues_indexes, nodes_features):


nodes_features,residues_indexes, c_alpha_positions, local_frame, position_local_frame, absolute_positions = get_node_features(structure)
pairwise_distances = utils.compute_distance_matrix(c_alpha_positions)
edges_features, nodes_neighb = get_edge_features(residues_indexes, pairwise_distances, nodes_features)
print(absolute_positions.shape)
edge_indexes = get_edge_index(nodes_neighb, edges_features)
all_features = {"nodes_features":nodes_features, "edges_features":edges_features,
                "N_residues":len(residues_indexes), "edge_indexes":edge_indexes, "local_frame":local_frame,
                "position_local_frame":position_local_frame, "absolute_positions":absolute_positions}


np.save(out_file, all_features)

