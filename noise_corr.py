import numpy as np
from numpy.linalg import matrix_rank
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


nfreq = 256
npix = 12 * 256**2
sigma = 2.0e-4 # make noise close to 21 cm signal

ns_map = sigma * np.random.randn(nfreq, npix)
ns_ns_corr = np.dot(ns_map, ns_map.T) / npix

name = 'ns_ns'
print 'Rank of %s: %d' % (name, matrix_rank(ns_ns_corr))

plt.figure()
plt.imshow(ns_ns_corr, origin='lower')
plt.colorbar()
plt.savefig('corr_figure/%s.png' % name)
