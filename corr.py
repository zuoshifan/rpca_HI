import numpy as np
from numpy.linalg import matrix_rank
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

ga_ga_corr = np.dot(ga_map, ga_map.T) / npix
ga_ps_corr = np.dot(ga_map, ps_map.T) / npix
ga_cm_corr = np.dot(ga_map, cm_map.T) / npix
ps_ps_corr = np.dot(ps_map, ps_map.T) / npix
ps_cm_corr = np.dot(ps_map, cm_map.T) / npix
cm_cm_corr = np.dot(cm_map, cm_map.T) / npix

fg_fg_corr = np.dot(fg_map, fg_map.T) / npix
fg_cm_corr = np.dot(fg_map, cm_map.T) / npix

tt_tt_corr = np.dot(tt_map, tt_map.T) / npix

diff_corr = tt_tt_corr - (fg_fg_corr + cm_cm_corr)

corrs = {
          'ga_ga': ga_ga_corr,
          'ga_ps': ga_ps_corr,
          'ga_cm': ga_cm_corr,
          'ps_ps': ps_ps_corr,
          'ps_cm': ps_cm_corr,
          'cm_cm': cm_cm_corr,
          'fg_fg': fg_fg_corr,
          'fg_cm': fg_cm_corr,
          'tt_tt': tt_tt_corr,
          'diff': diff_corr,
        }

for name, corr in corrs.items():
    print 'Rank of %s: %d' % (name, matrix_rank(corr))

    plt.figure()
    plt.imshow(corr, origin='lower')
    plt.colorbar()
    plt.savefig('%s.png' % name)
