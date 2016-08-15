from numpy.linalg import matrix_rank
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


with h5py.File('corr_data/corr.hdf5', 'r') as f:
    for name, dset in f.iteritems():
        corr = dset[:]
        print 'Rank of %s: %d' % (name, matrix_rank(corr))

        plt.figure()
        plt.imshow(corr, origin='lower')
        plt.colorbar()
        plt.savefig('figure/%s.png' % name)
