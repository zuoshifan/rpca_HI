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
R_HInh = np.dot(U*(1.0/s)**0.5, VT) # R_HI^-1/2

U1, s1, V1T = la.svd(np.dot(np.dot(R_HInh, R_tt), R_HInh))

# plot eigen values
# plt.figure()
# plt.semilogy(range(len(s1)), s1)
# plt.semilogy(range(len(s1)), s1, 'ro', markersize=4.0)
# plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
# plt.xlim(-1, 256)
# plt.ylabel('Eigen-values')
# plt.savefig('reconstruct/eig_val.png')
# plt.clf()

# # minimizing the AIC
# AIC = np.zeros_like(s1)
# tmp = s1 - np.log(s1) - 1.0
# for m in range(len(s1)):
#     AIC[m] = 2.0*m + np.sum(tmp[m+1:])
# n = np.argmin(AIC)

cind = len(cm_map) / 2 # central frequency index
normalize = True # normalize cl to l(l+1)Cl/2pi

# threshold = [ 0.9 + i * 0.05 for i in range(16) ]
threshold = [ 0.9 + i * 0.1 for i in range(32) ] + [1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e8, 1.0e12] + [ 3500.0 ]
Ri = la.inv(R_tt)
for td in threshold:
    # reconstruct 21cm map
    n = np.where(s1<td)[0][0]
    S = np.dot(R_HIh, U1[:, n:])
    STRiS = np.dot(S.T, np.dot(Ri, S))
    W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
    rec_cm = np.dot(W, tt_map)

    fig = plt.figure(1, figsize=(13, 5))
    healpy.mollview(rec_cm[cind], fig=1, title='')
    healpy.graticule()
    fig.savefig('reconstruct/rec_cm_%.2f.png' % td)
    fig.clf()

    fig = plt.figure(1, figsize=(13, 5))
    healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='')
    healpy.graticule()
    fig.savefig('reconstruct/diff_%.2f.png' % td)
    fig.clf()

    # compute cl
    cl_sim = healpy.anafast(cm_map[cind])
    cl_est = healpy.anafast(rec_cm[cind])
    if normalize:
        l = np.arange(len(cl_sim))
        factor = l*(l + 1) / (2*np.pi)
        cl_sim *= factor
        cl_est *= factor

    plt.figure()
    plt.plot(cl_sim, label='Input HI')
    plt.plot(cl_est, label='Recovered HI')
    if normalize:
        plt.plot(cl_sim - cl_est, label='Residual')
    plt.legend()
    plt.xlabel(r'$l$')
    if normalize:
        plt.ylabel(r'$l(l+1) C_l^{TT}/2\pi$')
    else:
        plt.ylabel(r'$C_l^{TT}$')
    plt.savefig('reconstruct/cl_%.2f.png' % td)
    plt.clf()



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
