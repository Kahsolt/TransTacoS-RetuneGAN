#!/usr/bin/env python3
# Author: Armit
# Create Time: 2021/03/04 

from sys import argv
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import seaborn as sns

ID = (len(argv) >= 2 and argv[1] or str(randrange(10000) + 1)).rjust(6,'0')

plt.subplot(311)
plt.title('energy')
e = np.load('DataBaker.preproc/energy/DataBaker-energy-%s.npy' % ID)
plt.xlim(0, len(e))
plt.plot(e)

plt.subplot(312)
plt.title('f0')
f = np.load('DataBaker.preproc/f0/DataBaker-f0-%s.npy' % ID)
plt.xlim(0, len(f))
plt.plot(f)

plt.subplot(313)
plt.title('mel')
m = np.load('DataBaker.preproc/mel/DataBaker-mel-%s.npy' % ID)
sns.heatmap(m.T[::-1], cbar=False)

plt.show()
