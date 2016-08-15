import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


with h5py.File('decomp.hdf5', 'r') as f:
    tt_tt = f['tt_tt'][:]
    cm_cm = f['cm_cm'][:]
    L = f['L'][:]
    S = f['S'][:]

res = tt_tt - L - S
diff = cm_cm - S

plt.figure()
plt.subplot(321)
plt.imshow(tt_tt, origin='lower')
plt.colorbar()
plt.subplot(322)
plt.imshow(cm_cm, origin='lower')
plt.colorbar()
plt.subplot(323)
plt.imshow(L, origin='lower')
plt.colorbar()
plt.subplot(324)
plt.imshow(S, origin='lower')
plt.colorbar()
plt.subplot(325)
plt.imshow(res, origin='lower')
plt.colorbar()
plt.subplot(326)
plt.imshow(diff, origin='lower')
plt.colorbar()
plt.savefig('decomp.png')
