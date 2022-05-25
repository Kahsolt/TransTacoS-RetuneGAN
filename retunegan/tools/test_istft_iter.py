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

import hparam as hp

R.seed(114514)


def decompose(D):
    P, S = np.angle(D), np.abs(D)
    return P, S


dp = r'C:\Users\Kahsolt\Desktop\Workspace\Data\DataBaker\Wave'
fn = R.choice([fn for fn in os.listdir(dp) if fn.endswith('.wav')])
fp = os.path.join(dp, fn)

y = L.load(fp, hp.sample_rate)[0]
ylen = len(y)
D = L.stft(y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
P, S = decompose(D)

y_loss, P_loss, S_loss = [], [], []
D_i, P_i, S_i = D, P, S
for i in range(1000):
  #D_i = S_i * np.exp(1j * P_i)
  y_i = L.istft(D_i, win_length=hp.win_length, hop_length=hp.hop_length, length=ylen)
  D_i = L.stft(y_i, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
  P_i, S_i = decompose(D_i)

  y_l = np.mean(np.abs(y - y_i)) ; y_loss.append(y_l)
  P_l = np.mean(np.abs(P - P_i)) ; P_loss.append(P_l)
  S_l = np.mean(np.abs(S - S_i)) ; S_loss.append(S_l)

plt.subplot(3, 1, 1) ; plt.plot(y_loss)
plt.subplot(3, 1, 2) ; plt.plot(P_loss)
plt.subplot(3, 1, 3) ; plt.plot(S_loss)
plt.show()
