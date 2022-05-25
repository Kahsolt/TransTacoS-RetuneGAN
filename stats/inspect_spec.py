#!/usr/bin/env python3
# Author: Armit
# Create Time: 2021/03/23 

from sys import argv
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import seaborn as sns

ID = (len(argv) >= 2 and argv[1] or str(randrange(20) + 1)).rjust(6,'0')

plt.subplot(211)
plt.title('linear spec')
m = np.load('DataBaker.spec/databaker-spec-%s.npy' % ID)
sns.heatmap(m.T[::-1], cbar=False)

plt.subplot(212)
plt.title('mel spec')
m = np.load('DataBaker.spec/databaker-mel-%s.npy' % ID)
sns.heatmap(m.T[::-1], cbar=False)

plt.show()
