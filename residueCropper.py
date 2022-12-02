import numpy as np
import Bio.PDB as bpdb
from Bio.PDB import PDBParser


class ResSelect(bpdb.Select):
    def __init__(self, constraints):
        super(ResSelect, self).__init__()
        self.constraints = constraints

    def accept_residue(self, res):
        if res.parent.id in self.constraints.keys():
            condition_low = self.constraints[res.parent.id]["start_residues"] <= res.id[1]
            condition_high = res.id[1] <= self.constraints[res.parent.id]["end_residues"]
            condition = condition_low & condition_high
            if condition.any():
                return False
            else:
                return True

        return True


file = "ranked_0.pdb"
out_file = "data/ranked_0_helixTrimmed.npy"
parser = PDBParser(PERMISSIVE=0)

structure = parser.get_structure("A", file)
start_residues = [471, 448]
end_residues = [475, 452]
chain_id = ["A", "B"]
constraints = {"A":{"start_residues":np.array([471]), "end_residues":np.array([475])},
               "B":{"start_residues":np.array([448]), "end_residues":np.array([452])}}


io = bpdb.PDBIO()
io.set_structure(structure)
io.save("ranked_0_helixTrimmed.npy", ResSelect(constraints))





