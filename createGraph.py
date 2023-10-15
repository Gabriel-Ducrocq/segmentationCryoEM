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


#file = "data/ranked_0_round1.pdb"
file = "../VAEProtein/data/MD_dataset/test_10000.pdb"
out_file = "../VAEProtein/data/vaeContinuousMD/features_open.npy"
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
                residues_indexes.append(N_residue)
                name = residue.get_resname()
                if name != "LBV":
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


#def get_data_x(residues_indexes, nodes_features):


nodes_features,residues_indexes, c_alpha_positions, local_frame, position_local_frame, absolute_positions = get_node_features(structure)
all_features = {"nodes_features":nodes_features,  "N_residues":len(residues_indexes), "local_frame":local_frame,
                "position_local_frame":position_local_frame, "absolute_positions":absolute_positions}


np.save(out_file, all_features)

