#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/03/10

# use this script to decide `hp.envelope_pool_k`
# higher sample_rate needs larger `k`


import os
import random as R
import librosa as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import hparam as hp


max_pools = [
    nn.MaxPool1d( 64, 1),
    nn.MaxPool1d(128, 1),   # for 16000 Hz
    nn.MaxPool1d(160, 1),   # for 22050 Hz
    nn.MaxPool1d(256, 1),   # for 44100 Hz
    nn.MaxPool1d(512, 1),
]

dp = r'C:\Users\Kahsolt\Desktop\Workspace\Data\DataBaker\Wave'
fn = R.choice([fn for fn in os.listdir(dp) if fn.endswith('.wav')])
fp = os.path.join(dp, fn)


y = L.load(fp, hp.sample_rate)[0]
y_np = y
y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)

plt.subplot(4, 1, 1)
plt.plot(y.numpy().squeeze())
plt.title('y')
pool = max_pools[2]
u =  pool( y)
d = -pool(-y)
plt.subplot(4, 1, 2)
plt.title('y_envolope')
plt.plot(u.numpy().squeeze())
plt.plot(d.numpy().squeeze())
plt.subplot(4, 1, 3)
plt.title('y_even')
plt.plot(y_np[::2])
plt.subplot(4, 1, 4)
plt.title('y_odd')
plt.plot(y_np[1::2])
plt.show()

exit(0)

y = L.load(fp, hp.sample_rate)[0]
y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)

plt.subplot(len(max_pools)+1, 1, 1)
plt.plot(y.numpy().squeeze())
for i, pool in enumerate(max_pools):
    u =  pool( y)
    d = -pool(-y)
    plt.subplot(len(max_pools)+1, 1, i+2)
    plt.plot(u.numpy().squeeze())
    plt.plot(d.numpy().squeeze())
plt.show()
