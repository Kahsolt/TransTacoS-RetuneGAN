#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/03/10

# use this script to decide `hp.downsample_pool_k`
# higher sample_rate needs larger `k`


import os
import random as R
import librosa as L
from scipy.io import wavfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio.transforms as T

import hparam as hp

resamplers = [
    T.Resample(hp.sample_rate, hp.sample_rate//2, resampling_method='sinc_interpolation'), # kaiser_window
    T.Resample(hp.sample_rate, hp.sample_rate//4, resampling_method='sinc_interpolation'),
    T.Resample(hp.sample_rate, hp.sample_rate//8, resampling_method='sinc_interpolation'),
]
avg_pools = [
    nn.AvgPool1d(kernel_size=2, stride=2, padding=1),
    nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
    nn.AvgPool1d(kernel_size=6, stride=2, padding=3),  # for 16000 Hz
    nn.AvgPool1d(kernel_size=8, stride=2, padding=4),
]

dp = r'C:\Users\Kahsolt\Desktop\Workspace\Data\DataBaker\Wave'
fn = R.choice([fn for fn in os.listdir(dp) if fn.endswith('.wav')])
fp = os.path.join(dp, fn)

y = L.load(fp, hp.sample_rate)[0]
y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)

plt.subplot(len(resamplers)+1, 1, 1)
plt.plot(y.numpy().squeeze())
for i, resampler in enumerate(resamplers):
    s = resampler(y)
    plt.subplot(len(resamplers)+1, 1, i+2)
    plt.plot(s.numpy().squeeze())
plt.show()

for i, avg_pool in enumerate(avg_pools):
    plt.subplot(len(resamplers)+1, 1, 1)
    plt.plot(y.numpy().squeeze())
    s = y
    for j in range(len(resamplers)):
        s = avg_pool(s)
        plt.subplot(len(resamplers)+1, 1, j+2)
        plt.plot(s.numpy().squeeze())
    plt.show()

