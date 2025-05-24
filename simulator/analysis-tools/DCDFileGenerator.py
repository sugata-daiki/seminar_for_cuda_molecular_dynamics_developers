import numpy as np
import mdtraj as md
import sys
import warnings

args = sys.argv
warnings.simplefilter('ignore')

TrajFile = args[1]
lx = float(args[2])
ly = float(args[3])
lz = float(args[4])

lines = []

with open(TrajFile + '.dat') as f:
    lines = f.read().splitlines()


for i in range(len(lines)):
    lines[i] = np.array(lines[i].split(), dtype=float)

raw_data = np.array(lines, dtype=object)

n_frames = len(raw_data)
n_atoms_in_final_data = int(len(raw_data[-1])/3)


print("number of particles :", n_atoms_in_final_data)
print("number of frames :", n_frames)
print("PBC size:", lx, ly, lz)

tmpx = np.full((n_frames, n_atoms_in_final_data), -100.0);
tmpy = np.full((n_frames, n_atoms_in_final_data), -100.0);
tmpz = np.full((n_frames, n_atoms_in_final_data), -100.0)

cell_length = np.zeros((n_frames, 3))
cell_angle = np.full((n_frames, 3), 90.0)

for i in range(n_frames):
    n_atoms_in_frame_i = int(len(raw_data[i])/3)
    tmpx[i, :n_atoms_in_frame_i] = np.array(raw_data[i][0:n_atoms_in_frame_i], dtype = 'float')
    tmpy[i, :n_atoms_in_frame_i] = np.array(raw_data[i][n_atoms_in_frame_i:2*n_atoms_in_frame_i], dtype = 'float')
    tmpz[i, :n_atoms_in_frame_i] = np.array(raw_data[i][2*n_atoms_in_frame_i:], dtype = 'float')

    cell_length[i, 0] = lx
    cell_length[i, 1] = ly
    cell_length[i, 2] = lz


data = np.array((tmpx, tmpy, tmpz), dtype = np.float64).transpose(1, 2, 0)
with md.formats.DCDTrajectoryFile(TrajFile + '.dcd', 'w') as f:
    f.write(data, cell_length, cell_angle)
