import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
import healpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


out_dir = './reconstruct/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

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

# with h5py.File('decomp_PCP.hdf5', 'r') as f:
with h5py.File('decomp/decomp.hdf5', 'r') as f:
    R_tt = f['tt_tt'][:]
    R_HI = f['S'][:]

s, U = la.eigh(R_HI)
R_HIh = np.dot(U*s**0.5, U.T)
R_HInh = np.dot(U*(1.0/s)**0.5, U.T)
s1, U1 = la.eigh(np.dot(np.dot(R_HInh, R_tt), R_HInh))
# print s1[::-1]

# # plot eigen values
# plt.figure()
# plt.semilogy(range(len(s1)), s1[::-1])
# plt.semilogy(range(len(s1)), s1[::-1], 'ro', markersize=4.0)
# plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
# plt.xlim(-1, 256)
# plt.ylabel('Eigen-values')
# plt.savefig(out_dir + 'eig_val.png')
# plt.clf()

cind = len(cm_map) / 2 # central frequency index

# threshold = [ 1.0, 1.05, 1.1, 1.15, 1.2, 5.0e3 ]
threshold = [ 1.0, 1.1, 1.2, 5.0e3 ]

Ri = la.inv(R_tt)
for td in threshold:
    # reconstruct 21cm map
    n = np.where(s1>td)[0][0]
    S = np.dot(R_HIh, U1[:, :n])
    STRiS = np.dot(S.T, np.dot(Ri, S))
    W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
    rec_cm = np.dot(W, tt_map)

    # plot reconstructed 21cm map
    fig = plt.figure(1, figsize=(13, 5))
    healpy.mollview(rec_cm[cind], fig=1, title='')
    healpy.graticule()
    fig.savefig(out_dir + 'rec_cm_%.2f.png' % td)
    fig.clf()

    # plot difference map
    fig = plt.figure(1, figsize=(13, 5))
    healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='')
    healpy.graticule()
    fig.savefig(out_dir + 'diff_%.2f.png' % td)
    fig.clf()

    # compute cl
    cl_sim = healpy.anafast(cm_map[cind])
    cl_est = healpy.anafast(rec_cm[cind])

    # plot cl
    plt.figure()
    plt.plot(cl_sim, label='Input HI')
    plt.plot(cl_est, label='Recovered HI')
    if td > 10.0:
        plt.ylim(0, 1.0e-10)
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{TT}$')
    plt.savefig(out_dir + 'cl_%.2f.png' % td)
    plt.clf()

    # normalize cl to l(l+1)Cl/2pi
    l = np.arange(len(cl_sim))
    factor = l*(l + 1) / (2*np.pi)
    cl_sim *= factor
    cl_est *= factor

    # plot normalized cl
    plt.figure()
    plt.plot(cl_sim, label='Input HI')
    plt.plot(cl_est, label='Recovered HI')
    plt.plot(cl_sim - cl_est, label='Residual')
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1) C_l^{TT}/2\pi$')
    plt.savefig(out_dir + 'cl_normalize_%.2f.png' % td)
    plt.clf()




# U, s, VT = la.svd(R_HI)
# R_HIh = np.dot(U*s**0.5, VT) # R_HI^1/2
# R_HInh = np.dot(U*(1.0/s)**0.5, VT) # R_HI^-1/2

# U1, s1, V1T = la.svd(np.dot(np.dot(R_HInh, R_tt), R_HInh))

# # plot eigen values
# plt.figure()
# plt.semilogy(range(len(s1)), s1[::-1])
# plt.semilogy(range(len(s1)), s1[::-1], 'ro', markersize=4.0)
# plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
# plt.xlim(-1, 256)
# plt.ylabel('Eigen-values')
# # plt.savefig('reconstruct/eig_val.png')
# plt.savefig('eig_val.png')
# plt.clf()
# err

# # minimizing the AIC
# AIC = np.zeros_like(s1)
# tmp = s1 - np.log(s1) - 1.0
# for m in range(len(s1)):
#     AIC[m] = 2.0*m + np.sum(tmp[m+1:])
# n = np.argmin(AIC)

# cind = len(cm_map) / 2 # central frequency index
# normalize = True # normalize cl to l(l+1)Cl/2pi

# # threshold = [ 0.9 + i * 0.05 for i in range(16) ]
# threshold = [ 0.9 + i * 0.1 for i in range(32) ] + [1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e8, 1.0e12] + [ 3500.0 ]
# Ri = la.inv(R_tt)
# for td in threshold:
#     # reconstruct 21cm map
#     n = np.where(s1<td)[0][0]
#     S = np.dot(R_HIh, U1[:, n:])
#     STRiS = np.dot(S.T, np.dot(Ri, S))
#     W = np.dot(np.dot(np.dot(S, la.inv(STRiS)), S.T), Ri)
#     rec_cm = np.dot(W, tt_map)

#     fig = plt.figure(1, figsize=(13, 5))
#     healpy.mollview(rec_cm[cind], fig=1, title='')
#     healpy.graticule()
#     fig.savefig('reconstruct/rec_cm_%.2f.png' % td)
#     fig.clf()

#     fig = plt.figure(1, figsize=(13, 5))
#     healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='')
#     healpy.graticule()
#     fig.savefig('reconstruct/diff_%.2f.png' % td)
#     fig.clf()

#     # compute cl
#     cl_sim = healpy.anafast(cm_map[cind])
#     cl_est = healpy.anafast(rec_cm[cind])
#     if normalize:
#         l = np.arange(len(cl_sim))
#         factor = l*(l + 1) / (2*np.pi)
#         cl_sim *= factor
#         cl_est *= factor

#     plt.figure()
#     plt.plot(cl_sim, label='Input HI')
#     plt.plot(cl_est, label='Recovered HI')
#     if normalize:
#         plt.plot(cl_sim - cl_est, label='Residual')
#     plt.legend()
#     plt.xlabel(r'$l$')
#     if normalize:
#         plt.ylabel(r'$l(l+1) C_l^{TT}/2\pi$')
#     else:
#         plt.ylabel(r'$C_l^{TT}$')
#     plt.savefig('reconstruct/cl_%.2f.png' % td)
#     plt.clf()
