import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
import healpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



with h5py.File('corr_data/corr.hdf5', 'r') as f:
    cm_cm_corr = f['cm_cm'][:]
    tt_tt_corr = f['tt_tt'][:]

# from r_pca import R_pca
# rpca = R_pca(tt_tt_corr, mu=1.0e7, lmbda=None)
# L, S = rpca.fit(tol=1.0e-14, max_iter=20000, iter_print=100)

from noncvx_rpca import rpca
L, S, res = rpca(tt_tt_corr, mu=1.0e7, gamma=1.0, norm_type='1')

# print res
print matrix_rank(L)
print matrix_rank(S)
print la.norm(cm_cm_corr - S, ord='fro') / la.norm(cm_cm_corr, ord='fro')

with h5py.File('decomp.hdf5', 'w') as f:
    f.create_dataset('tt_tt', data=tt_tt_corr)
    f.create_dataset('cm_cm', data=cm_cm_corr)
    f.create_dataset('L', data=L)
    f.create_dataset('S', data=S)
