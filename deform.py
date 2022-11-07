import numpy as np
from Bio.PDB.PDBParser import PDBParser


def global_translation(base_structure, translation_vector):
    rotation_matrix = np.eye(3,3)
    structure = base_structure.copy()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.transform(rotation_matrix, translation_vector)

                #name = residue.get_resname()
                #residue["CA"].set_coord(residue["CA"].get_coord() + translation_vector)
                #residue["N"].set_coord(residue["N"].get_coord() + translation_vector)
                #if name == "GLY":
                #    residue["C"].set_coord(residue["C"].get_coord() + translation_vector)
                #else:
                #    residue["C"].set_coord(residue["C"].get_coord() + translation_vector)

    return structure


def sample_global_translations(structure, n_deformations = 100, sigma = 2, sigma_perturb=0.5):
    all_structures = []
    latent_variables = []
    translation_vectors = []
    for i in range(n_deformations):
        print(i)
        latent_variable = sigma*np.random.normal(size=3)
        translation_vector = latent_variable + sigma_perturb*np.random.normal(size=3)

        struct = global_translation(structure, translation_vector)
        all_structures.append(struct)
        translation_vectors.append(translation_vector)

    return all_structures, latent_variables, translation_vectors


def check(base_struct, all_structures, latent_variables, translation_vectors):
    base_residues = []
    all_diffs_x = []
    all_diffs_y = []
    all_diffs_z = []
    for model in base_struct:
        for chain in model:
            for residue in chain:
                base_residues.append(residue)

    for i in range(len(all_structures)):
        res_num = 0
        struct = all_structures[i]
        translation_vector = translation_vectors[i]
        for model in struct:
            for chain in model:
                for residue in chain:
                    name = residue.get_resname()
                    x = residue["CA"].get_coord() - translation_vector
                    y = residue["N"].get_coord() - translation_vector
                    if name == "GLY":
                        z = residue["C"].get_coord() - translation_vector
                    else:
                        z = residue["C"].get_coord() - translation_vector

                    base_res = base_residues[res_num]
                    base_x = base_res["CA"].get_coord()
                    base_y = base_res["N"].get_coord()
                    base_z = base_res["C"].get_coord()

                    diff_x = base_x - x
                    all_diffs_x.append(diff_x)
                    diff_y = base_y - y
                    all_diffs_y.append(diff_y)
                    diff_z = base_z - z
                    all_diffs_z.append(diff_z)

                    res_num += 1

    return all_diffs_x, all_diffs_y, all_diffs_z







file = "data/ranked_0_round1.pdb"
out_file = "data/deformed_structures.npy"
parser = PDBParser(PERMISSIVE=0)

structure= parser.get_structure("A", file)


all_structures, latent_variables, translation_vectors = sample_global_translations(structure)

all_x, all_y, all_z = check(structure, all_structures, latent_variables, translation_vectors)

dataset = {"all_structures":all_structures, "latent_variables":latent_variables,
           "translation_vectors": translation_vectors}


np.save(out_file, dataset)


