import mdtraj as md
import numpy as np
from mdtraj.core.topology import Topology
import sys

def create_dummy_pdb(num_atoms, output_pdb):
    top = Topology()
    chain = top.add_chain()
    residue = top.add_residue('DUM', chain)

    for i in range(num_atoms):
        top.add_atom(f'X{i}', element=md.element.carbon, residue=residue)

    positions = np.zeros((1, num_atoms, 3))

    traj = md.Trajectory(xyz=positions, topology=top)

    traj.save_pdb(output_pdb)
    print(f"Dummy PDB saved as: {output_pdb}")

args = sys.argv
create_dummy_pdb(int(args[1]), args[2])
