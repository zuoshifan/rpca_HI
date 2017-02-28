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

# load corr data
with h5py.File('corr_data/corr.hdf5', 'r') as f:
    cm_cm = f['cm_cm'][:]
    fg_fg = f['fg_fg'][:]
    tt_tt = f['tt_tt'][:]


# simultaneously diagonalize fg_fg and cm_cm
# s, V = la.eigh(cm_cm, fg_fg)
s, V = la.eigh(fg_fg, cm_cm)

# plot eig-vals
plt.figure()
plt.semilogy(s[::-1])
plt.semilogy(s[::-1], 'ro')
plt.savefig(out_dir + 'eigval.png')
plt.close()


# # simultaneously diagonalize fg_fg and cm_cm
# s, V = la.eigh(fg_fg, cm_cm)
# # print np.dot(np.dot(V.T, cm_cm), V)
# # print np.diag(np.dot(np.dot(V.T, fg_fg), V))
# # print matrix_rank(fg_fg), matrix_rank(cm_cm)
# # print s
# # s, U = la.eigh(fg_fg)
# # U, s, VT = la.svd(fg_fg)
# # print s

# # # Cholesky decomposition of tt_tt
# # # U = la.cholesky(tt_tt, lower=False)
# # U = la.sqrtm(tt_tt)
# # print U
# # # print np.allclose(U, U.T)
# # # print np.allclose(np.dot(U.T, U), tt_tt)

# # U1 = la.inv(np.dot(U, V).T)
# # s_est = np.dot(U1, np.dot(V.T, tt_map))
# # print s_est.shape

# U = la.sqrtm(np.dot(np.dot(V.T, fg_fg), V))
# s_est = np.dot(U.T, np.dot(V.T, tt_map))

# cind = s_est.shape[0] / 2
# # plot s_est
# plt.figure()
# fig = plt.figure(1, figsize=(13, 5))
# healpy.mollview(s_est[cind], fig=1, title='')
# healpy.graticule()
# fig.savefig('s_est.png')
# plt.close()
