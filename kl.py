import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
import healpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


out_dir = './kl/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# ladd sky map
map_dir = 'sky_map/'
ps_name = map_dir + 'sim_pointsource_256_700_800_256.hdf5'
ga_name = map_dir + 'sim_galaxy_256_700_800_256.hdf5'
cm_name = map_dir + 'sim_21cm_256_700_800_256.hdf5'
with h5py.File(ps_name, 'r') as f:
    ps_map = f['map'][:, 0, :]
with h5py.File(ga_name, 'r') as f:
    ga_map = f['map'][:, 0, :]
with h5py.File(cm_name, 'r') as f:
    cm_map = f['map'][:, 0, :]

fg_map = ps_map + ga_map
tt_map = fg_map + cm_map # total signal

with h5py.File('decomp/decomp.hdf5', 'r') as f:
    R_HI = f['S'][:]
    R_f = f['L'][:]


# simultaneously diagonalize R_f and R_HI
s, V = la.eigh(R_HI, R_f)
# s, V = la.eigh(R_f, R_HI)

# plot eig-vals
plt.figure()
plt.semilogy(s[::-1])
plt.semilogy(s[::-1], 'ro')
plt.savefig(out_dir + 'eigval.png')
plt.close()
