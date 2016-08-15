import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
import healpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

with h5py.File('decomp_PCP.hdf5', 'r') as f:
    R_tt = f['tt_tt'][:]
    R_HI = f['S'][:]

U, s, VT = la.svd(R_HI)
R_HIh = np.dot(U*s**0.5, VT) # R_HI^1/2
R_HInh = np.dot(U*(1.0/s**0.5), VT) # R_HI^-1/2

U1, s1, V1T = la.svd(np.dot(np.dot(R_HInh, R_tt), R_HInh))

# # minimizing the AIC
# AIC = np.zeros_like(s1)
# tmp = s1 - np.log(s1) - 1.0
# for m in range(len(s1)):
#     AIC[m] = 2.0*m + np.sum(tmp[m+1:])
# n = np.argmin(AIC)

# n = np.where(s1<1.1)[0][0]
n = np.where(s1<1.15)[0][0]
S = np.dot(R_HIh, U1[:, n:])
Ri = la.inv(R_tt)
STRiS = np.dot(S.T, np.dot(Ri, S))
W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
rec_cm = np.dot(W, tt_map)
fig = plt.figure(1, figsize=(13, 5))
healpy.mollview(rec_cm[0], fig=1, title='')
healpy.graticule()
fig.savefig('rec_cm.png')
fig.clf()

plt.figure()
cl_est = healpy.anafast(rec_cm[0])
cl_sim = healpy.anafast(cm_map[0])
plt.plot(cl_est, label='est')
plt.plot(cl_sim, label='sim')
plt.legend()
plt.savefig('cl.png')



# s2, U1 = la.eigh(cm_cm_corr)
# s2 = s2[::-1]
# U1 = U1[:, ::-1]
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
# fig.clf()



# # svd of the total signal
# gtt = np.asfortranarray(tt_map)
# dtt = core.DistributedMatrix.from_global_array(gtt, rank=0)
# U, s_tt, VT = rt.svd(dtt)
# if rank == 0:
#     print s_tt
# # gU = U.to_global_array(rank=0)
# # gVT = VT.to_global_array(rank=0)

# # VT = np.dot(1.0/s_tt*U1.T, tt_map) # the actual V^T
# VT = np.dot(np.dot(np.diag(1.0/s_tt), U1.T), tt_map) # the actual V^T

# if rank == 0:
#     rec_cm = np.dot(U1*s2**0.5, VT)
#     print rec_cm.shape

#     fig = plt.figure(1, figsize=(13, 5))
#     healpy.mollview(rec_cm[0], fig=1, title='')
#     healpy.graticule()
#     # fig.savefig('rec_cm.png')
#     fig.savefig('rec_cm5.png')
#     fig.clf()
