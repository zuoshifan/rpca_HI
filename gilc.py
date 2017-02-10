import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
from scipy import optimize
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
    L = f['L'][:]

with h5py.File('corr_data/corr.hdf5', 'r') as f:
    R_f = f['fg_fg'][:]

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
# plt.close()

# plot difference of foreground coherence
plt.figure()
# plt.subplot(221)
# plt.imshow(R_f)
# plt.colorbar()
# plt.subplot(222)
# plt.imshow(L)
# plt.colorbar()
# plt.subplot(223)
plt.imshow(R_f - L)
plt.colorbar()
plt.savefig(out_dir + 'Rf_diff.png')
plt.close()

# Equation for Gaussian
def f(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))

bins = 201
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
    plt.close()

    # plot difference map
    fig = plt.figure(1, figsize=(13, 5))
    healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='')
    healpy.graticule()
    fig.savefig(out_dir + 'diff_%.2f.png' % td)
    plt.close()

    # plot scatter
    plt.figure()
    plt.scatter(cm_map[cind], rec_cm[cind])
    plt.xlim(-0.002, 0.002)
    plt.ylim(-0.002, 0.002)
    ref_line = np.linspace(-0.002, 0.002, 100)
    plt.plot(ref_line, ref_line, 'k--')
    plt.savefig(out_dir + 'scatter_%.2f.png' % td)
    plt.close()

    # plot hist
    plt.figure()
    data = plt.hist(rec_cm[cind]/cm_map[cind]-1, bins=bins, range=[-3, 3])
    plt.xlabel('recover/input' + r'${} - 1$')

    if td < 1.5:
        # Generate data from bins as a set of points
        x = [0.5 * (data[1][ii] + data[1][ii+1]) for ii in xrange(len(data[1])-1)]
        y = data[0]

        popt, pcov = optimize.curve_fit(f, x, y)
        a, b, c = popt

        xmax = max(abs(x[0]), abs(x[-1]))
        x_fit = np.linspace(-xmax, xmax, bins)
        y_fit = f(x_fit, *popt)

        lable = r'$a \, \exp{(- \frac{(x - \mu)^2} {2 \sigma^2})}$' + '\n\n' + r'$a = %f$' % a + '\n' + r'$\mu = %f$' % b + '\n' + r'$\sigma = %f$' % np.abs(c)
        plt.plot(x_fit, y_fit, lw=2, color="r", label=lable)
        plt.xlim(-xmax, xmax)
        plt.legend()

    plt.savefig(out_dir + 'hist_%.2f.png' % td)
    plt.close()



    # compute cl
    cl_sim = healpy.anafast(cm_map[cind])
    cl_est = healpy.anafast(rec_cm[cind])
    cl_simxest = healpy.anafast(cm_map[cind], rec_cm[cind])

    # plot cl
    plt.figure()
    plt.plot(cl_sim, label='Input HI')
    plt.plot(cl_est, label='Recovered HI')
    plt.plot(cl_simxest, label='cross', color='magenta')
    if td > 10.0:
        plt.ylim(0, 1.0e-10)
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{TT}$')
    plt.savefig(out_dir + 'cl_%.2f.png' % td)
    plt.close()

    # plot cross cl
    plt.figure()
    plt.plot(cl_simxest)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{TT, cross}$')
    plt.savefig(out_dir + 'xcl_%.2f.png' % td)
    plt.close()

    # plot transfer function cl_out / cl_in
    plt.figure()
    plt.plot(cl_est/cl_sim)
    plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
    plt.ylim(0, 2)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$T_l$')
    plt.savefig(out_dir + 'Tl_%.2f.png' % td)
    plt.close()

    # normalize cl to l(l+1)Cl/2pi
    l = np.arange(len(cl_sim))
    factor = l*(l + 1) / (2*np.pi)
    cl_sim *= factor
    cl_est *= factor
    cl_simxest *= factor

    # plot normalized cl
    plt.figure()
    plt.plot(cl_sim, label='Input HI')
    plt.plot(cl_est, label='Recovered HI')
    plt.plot(cl_simxest, label='cross', color='magenta')
    plt.plot(cl_sim - cl_est, label='Residual', color='red')
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1) C_l^{TT}/2\pi$')
    plt.savefig(out_dir + 'cl_normalize_%.2f.png' % td)
    plt.close()

    # plot normalized cross cl
    plt.figure()
    plt.plot(cl_simxest)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$l(l+1) C_l^{TT, clross}/2\pi$')
    plt.savefig(out_dir + 'xcl_normalize_%.2f.png' % td)
    plt.close()





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
