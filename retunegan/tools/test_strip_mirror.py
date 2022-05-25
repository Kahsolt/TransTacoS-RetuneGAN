#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/03/10

import os
import random as R
import librosa as L
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn

import hparam as hp

#R.seed(114514)


def decompose(D):
    P, S = np.angle(D), np.abs(D)
    logS = np.log(S.clip(1e-5, None))
    return P, S, logS


dp = r'C:\Users\Kahsolt\Desktop\Workspace\Data\DataBaker\Wave'
fn = R.choice([fn for fn in os.listdir(dp) if fn.endswith('.wav')])
fp = os.path.join(dp, fn)

sr, hsr = hp.sample_rate, hp.sample_rate//2
y = L.load(fp, hp.sample_rate)[0]
if len(y) % 2 != 0: y = y[:-1]
ylen = len(y)

avgpool = nn.AvgPool1d(hp.downsample_pool_k, 2)
Ty = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
for i in range(3):
    if Ty.shape[-1] % 2 != 0: Ty = Ty[:,:,:-1]
    even, odd = Ty[:,:,::2], Ty[:,:,1::2]
    diff = even - odd
    print(f'strip_mirror_loss({i}):', torch.mean(torch.abs(diff)).item())
    Ty = avgpool(Ty)


even, odd = y[::2], y[1::2]
mean = (even + odd) / 2
diff = even - odd
print('strip_mirror_loss:', np.mean(np.abs(diff)))

if False:
  plt.subplot(4, 1, 1) ; plt.plot(y)
  plt.subplot(4, 1, 2) ; plt.plot(even)
  plt.subplot(4, 1, 3) ; plt.plot(odd)
  plt.subplot(4, 1, 4) ; plt.plot(diff)
  plt.show()

if False:
    wavfile.write('y.wav',     sr, y)
    wavfile.write('even.wav', hsr, even)
    wavfile.write('odd.wav',  hsr, odd)
    wavfile.write('diff.wav', hsr, diff)

De = L.stft(even, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
Do = L.stft(odd,  n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
Dd = L.stft(diff, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
Dm = L.stft(mean, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
Pe, Se, logSe = decompose(De)
Po, So, logSo = decompose(Do)
Pd, Sd, logSd = decompose(Dd)
Pm, Sm, logSm = decompose(Dm)

print('|Pe - Po|:',       np.mean(np.abs(Pe - Po)))
print('|Se - So|:',       np.mean(np.abs(Se - So)))
print('|logSe - logSo|:', np.mean(np.abs(logSe - logSo)))
print()

print('|Pd - Pe|:',       np.mean(np.abs(Pd - Pe)))
print('|Sd - Se|:',       np.mean(np.abs(Sd - Se)))
print('|logSd - logSe|:', np.mean(np.abs(logSd - logSe)))
print('|Pd - Po|:',       np.mean(np.abs(Pd - Po)))
print('|Sd - So|:',       np.mean(np.abs(Sd - So)))
print('|logSd - logSo|:', np.mean(np.abs(logSd - logSo)))
print()


