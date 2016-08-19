import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
import healpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = './decomp/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with h5py.File('corr_data/corr.hdf5', 'r') as f:
    cm_cm_corr = f['cm_cm'][:]
    tt_tt_corr = f['tt_tt'][:]

from r_pca import R_pca
rpca = R_pca(tt_tt_corr, mu=1.0e8, lmbda=None)
L, S, err = rpca.fit(tol=1.0e-14, max_iter=20000, iter_print=100, return_err=True)

# from r_pca import MR_pca
# rpca = MR_pca(tt_tt_corr, mu=1.0e8, lmbda=None)
# L, S, err = rpca.fit(tol=1.0e-14, max_iter=20000, iter_print=100, return_err=True)

# from noncvx_rpca import rpca
# L, S, res = rpca(tt_tt_corr, mu=1.0e7, gamma=1.0, norm_type='1')

print err
print matrix_rank(L)
print matrix_rank(S)
print la.norm(cm_cm_corr - S, ord='fro') / la.norm(cm_cm_corr, ord='fro')
print np.allclose(L, L.T), np.allclose(S, S.T)
# sL, UL = la.eigh(L)
# sS, US = la.eigh(S)
# print sL
# print sS

with h5py.File(out_dir + 'decomp.hdf5', 'w') as f:
    f.create_dataset('tt_tt', data=tt_tt_corr)
    f.create_dataset('cm_cm', data=cm_cm_corr)
    f.create_dataset('L', data=L)
    f.create_dataset('S', data=S)
