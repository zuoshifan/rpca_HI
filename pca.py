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


out_dir = './pca_reconstruct/'
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

with h5py.File('decomp/decomp.hdf5', 'r') as f:
    R_tt = f['tt_tt'][:]
    R_HI = f['S'][:]

with h5py.File('corr_data/corr.hdf5', 'r') as f:
    R_f = f['fg_fg'][:]

# PCA for R_tt
s, U = la.eigh(R_tt)

xinds = range(len(s))

# plot eigen values
plt.figure()
plt.semilogy(xinds, s[::-1])
plt.semilogy(xinds, s[::-1], 'ro', markersize=4.0)
# plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
plt.xlim(-1, 256)
plt.ylabel('Eigen-values')
plt.savefig(out_dir + 'eig_val.png')
plt.close()

# plot eigen vectors
plt.figure()
plt.plot(xinds, U[:, -1])
plt.plot(xinds, U[:, -2])
plt.plot(xinds, U[:, -3])
plt.plot(xinds, U[:, -4])
plt.plot(xinds, U[:, -5])
plt.plot(xinds, U[:, -6])
plt.xlim(-1, 256)
plt.ylabel('Eigen-vector')
plt.savefig(out_dir + 'eig_vector.png')
plt.close()

# Equation for Gaussian
def f(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))

# Equation for Cauchy
def f1(x, a, b, c):
    return a * c/ ((x-b)**2 + c**2)

bins = 201
cind = len(cm_map) / 2 # central frequency index
# plot principal components
for i in xrange(1, 7):
    pc = np.dot(np.dot(U[:, -i], U[:, -i].T), tt_map)
    pc_sum = np.dot(np.dot(U[:, -i:], U[:, -i:].T), tt_map)
    res = tt_map - pc_sum

    # # plot pc
    # plt.figure()
    # fig = plt.figure(1, figsize=(13, 5))
    # healpy.mollview(pc[cind], fig=1, title='')
    # healpy.graticule(verbose=False)
    # fig.savefig(out_dir + 'pc_%d.png' % i)
    # plt.close()

    # plot pc_sum
    plt.figure()
    fig = plt.figure(1)
    healpy.mollview(pc_sum[cind], fig=1, title='', min=0, max=50)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'pc_sum_%d.png' % i)
    plt.close()

    # plot the difference covariance of pc_sum
    plt.figure()
    plt.imshow(R_f - np.dot(pc_sum, pc_sum.T)/pc_sum.shape[-1])
    plt.colorbar()
    plt.savefig(out_dir + 'Rf_diff_%d.png' % i)
    plt.close()

    # plot res
    plt.figure()
    fig = plt.figure(1)
    # healpy.mollview(res[cind], fig=1, title='')
    # healpy.mollview(res[cind], fig=1, title='', min=-0.001, max=0.001)
    # healpy.mollview(res[cind], fig=1, title='', min=-0.0005, max=0.0005)
    healpy.mollview(res[cind], fig=1, title='', min=-0.0004, max=0.0004)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'res_%d.png' % i)
    plt.close()

    rec_cm = res

    # plot difference map
    # fig = plt.figure(1, figsize=(13, 5))
    fig = plt.figure(1)
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='')
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.001, max=0.001)
    # healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.0005, max=0.0005)
    healpy.mollview(cm_map[cind] - rec_cm[cind], fig=1, title='', min=-0.0004, max=0.0004)
    healpy.graticule(verbose=False)
    fig.savefig(out_dir + 'diff_%d.png' % i)
    plt.close()

    # plot scatter
    plt.figure()
    plt.scatter(cm_map[cind], rec_cm[cind])
    plt.xlim(-0.002, 0.002)
    plt.ylim(-0.002, 0.002)
    ref_line = np.linspace(-0.002, 0.002, 100)
    plt.plot(ref_line, ref_line, 'k--')
    plt.savefig(out_dir + 'scatter_%d.png' % i)
    plt.close()

    # plot hist
    plt.figure()
    data = plt.hist(rec_cm[cind]/cm_map[cind]-1, bins=bins, range=[-3, 3])
    plt.xlabel('recover/input' + r'${} - 1$')

    if i >= 3:
        # Generate data from bins as a set of points
        x = [0.5 * (data[1][ii] + data[1][ii+1]) for ii in xrange(len(data[1])-1)]
        y = data[0]

        # popt, pcov = optimize.curve_fit(f, x, y)
        popt, pcov = optimize.curve_fit(f1, x, y)
        a, b, c = popt

        xmax = max(abs(x[0]), abs(x[-1]))
        x_fit = np.linspace(-xmax, xmax, bins)
        y_fit = f(x_fit, *popt)

        lable = r'$a \, \exp{(- \frac{(x - \mu)^2} {2 \sigma^2})}$' + '\n\n' + r'$a = %f$' % a + '\n' + r'$\mu = %f$' % b + '\n' + r'$\sigma = %f$' % np.abs(c)
        plt.plot(x_fit, y_fit, lw=2, color="r", label=lable)
        plt.xlim(-xmax, xmax)
        plt.legend()

        # # Generate data from bins as a set of points
        # x = [0.5 * (data[1][ii] + data[1][ii+1]) for ii in xrange(len(data[1])-1)]
        # y = data[0]

        # popt, pcov = optimize.curve_fit(f1, x, y)
        # a, b, c = popt

        # y_fit = f(x_fit, *popt)

        # lable = r'$a \, \exp{(- \frac{|x - \mu|} {c})}$' + '\n\n' + r'$a = %f$' % a + '\n' + r'$\mu = %f$' % b + '\n' + r'$\sigma = %f$' % np.abs(c)
        # plt.plot(x_fit, y_fit, lw=2, color="g", label=lable)
        # plt.legend()

    plt.savefig(out_dir + 'hist_%d.png' % i)
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
    plt.legend(loc='best')
    plt.xlabel(r'$l$')
    plt.ylabel(r'$C_l^{TT}$')
    plt.savefig(out_dir + 'cl_%d.png' % i)
    plt.close()

    # plot transfer function cl_out / cl_in
    plt.figure()
    plt.plot(cl_est/cl_sim)
    plt.axhline(y=1.0, linewidth=1.0, color='k', linestyle='--')
    plt.ylim(0, 2)
    plt.xlabel(r'$l$')
    plt.ylabel(r'$T_l$')
    plt.savefig(out_dir + 'Tl_%d.png' % i)
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
    plt.savefig(out_dir + 'cl_normalize_%d.png' % i)
    plt.close()
