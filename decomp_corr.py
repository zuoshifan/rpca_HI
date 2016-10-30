import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
import healpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from r_pca import R_pca

from mpi4py import MPI

from scalapy import core
import scalapy.routines as rt


comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

core.initmpi([2, 2], block_shape=[32, 32])



ps_name = 'sim_pointsource_256_700_800_256.hdf5'
ga_name = 'sim_galaxy_256_700_800_256.hdf5'
cm_name = 'sim_21cm_256_700_800_256.hdf5'
with h5py.File(ps_name, 'r') as f:
    ps_map = f['map'][:, 0, :]
with h5py.File(ga_name, 'r') as f:
    ga_map = f['map'][:, 0, :]
with h5py.File(cm_name, 'r') as f:
    cm_map = f['map'][:, 0, :]

fg_map = ps_map + ga_map
tt_map = fg_map + cm_map # total signal

npix = ps_map.shape[-1]

# ga_ga_corr = np.dot(ga_map, ga_map.T) / npix
# ga_ps_corr = np.dot(ga_map, ps_map.T) / npix
# ga_cm_corr = np.dot(ga_map, cm_map.T) / npix
# ps_ps_corr = np.dot(ps_map, ps_map.T) / npix
# ps_cm_corr = np.dot(ps_map, cm_map.T) / npix
cm_cm_corr = np.dot(cm_map, cm_map.T) / npix

# fg_fg_corr = np.dot(fg_map, fg_map.T) / npix
# fg_cm_corr = np.dot(fg_map, cm_map.T) / npix

tt_tt_corr = np.dot(tt_map, tt_map.T) / npix

rpca = R_pca(tt_tt_corr, mu=1.0e6, lmbda=None)
L, S = rpca.fit(tol=1.0e-14, max_iter=20000, iter_print=100)
print matrix_rank(L)
print matrix_rank(S)

# plt.figure()
# plt.subplot(221)
# plt.imshow(tt_tt_corr, origin='lower')
# plt.colorbar()
# plt.subplot(222)
# plt.imshow(tt_tt_corr-L-S, origin='lower')
# plt.colorbar()
# plt.subplot(223)
# plt.imshow(L, origin='lower')
# plt.colorbar()
# plt.subplot(224)
# plt.imshow(S, origin='lower')
# plt.colorbar()
# plt.savefig('LS.png')


# U1, s2, V1T = la.svd(S)
s2, U1 = la.eigh(cm_cm_corr)
s2 = s2[::-1]
U1 = U1[:, ::-1]
# U1, s2, V1T = la.svd(cm_cm_corr)
# C12 = np.dot(U1*s2**0.5, V1T)
# C_12 = np.dot(U1*(1.0/s2**0.5), V1T)
# U2, s22, V2T = la.svd(np.dot(np.dot(C_12, tt_tt_corr), C_12))
# n = np.where(s22<1.1)[0][0]
# S = np.dot(C12, U2[:, n:])
# R = tt_tt_corr
# Ri = la.inv(R)
# STRiS = np.dot(S.T, np.dot(Ri, S))
# W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
# rec_cm = np.dot(W, tt_map)
# fig = plt.figure(1, figsize=(13, 5))
# healpy.mollview(rec_cm[0], fig=1, title='')
# healpy.graticule()
# # fig.savefig('rec_cm.png')
# fig.savefig('rec_cm3.png')
# fig.close()



# svd of the total signal
gtt = np.asfortranarray(tt_map)
dtt = core.DistributedMatrix.from_global_array(gtt, rank=0)
U, s_tt, VT = rt.svd(dtt)
if rank == 0:
    print s_tt
# gU = U.to_global_array(rank=0)
# gVT = VT.to_global_array(rank=0)

# VT = np.dot(1.0/s_tt*U1.T, tt_map) # the actual V^T
VT = np.dot(np.dot(np.diag(1.0/s_tt), U1.T), tt_map) # the actual V^T

if rank == 0:
    rec_cm = np.dot(U1*s2**0.5, VT)
    print rec_cm.shape

    fig = plt.figure(1, figsize=(13, 5))
    healpy.mollview(rec_cm[0], fig=1, title='')
    healpy.graticule()
    # fig.savefig('rec_cm.png')
    fig.savefig('rec_cm5.png')
    fig.close()
