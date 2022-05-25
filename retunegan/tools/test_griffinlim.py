#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/03/10

import os
import random as R
import copy
import librosa as L
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import hparam as hp

R.seed(114514)


def decompose(D):
    P, S = np.angle(D), np.abs(D)
    logS = np.log(S.clip(1e-5, None))
    return P, S, logS


def griffin_lim(S, n_iter=60):
  X_best = copy.deepcopy(S)
  for _ in range(n_iter):
    X_t = L.istft(X_best, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
    X_best = L.stft(X_t, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
    phase = X_best / np.maximum(1e-8, np.abs(X_best))
    X_best = S * phase
  X_t = L.istft(X_best, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
  y = np.real(X_t)
  return y


def griffinlim(S, n_iter=60):
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  
  for _ in range(n_iter):
    full = np.abs(S).astype(np.complex) * angles
    inverse = L.istft(full, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
    rebuilt = L.stft(inverse, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
    angles = np.exp(1j * np.angle(rebuilt))
  full = np.abs(S).astype(np.complex) * angles
  inverse = L.istft(full, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
  return inverse


def griffinlim_conj(P, n_iter=60):
  #S = np.random.uniform(low=np.exp(-4), high=np.exp(4), size=P.shape)
  S = np.random.normal(loc=-4, scale=1, size=P.shape)
  #S = np.random.rand(*P.shape)
  
  for _ in range(n_iter):
    D = S * np.exp(1j * P)
    y_hat = L.istft(D, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
    D_hat = L.stft(y_hat, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
    S = np.abs(D_hat)

  D = S * np.exp(1j * P)
  y = L.istft(D, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
  return y


dp = r'C:\Users\Kahsolt\Desktop\Workspace\Data\DataBaker\Wave'
fn = R.choice([fn for fn in os.listdir(dp) if fn.endswith('.wav')])
fp = os.path.join(dp, fn)

y = L.load(fp, hp.sample_rate)[0]
ylen = len(y)
D = L.stft(y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
P, S, logS = decompose(D)

y1 = griffin_lim(S)
y2 = griffinlim(S)
y3 = griffinlim_conj(P)

plt.subplot(4, 1, 1) ; plt.plot(y)
plt.subplot(4, 1, 2) ; plt.plot(y1)
plt.subplot(4, 1, 3) ; plt.plot(y2)
plt.subplot(4, 1, 4) ; plt.plot(y3)
plt.show()

D1 = L.stft(y1, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
D2 = L.stft(y2, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
D3 = L.stft(y3, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
_, _, logS1 = decompose(D1)
_, _, logS2 = decompose(D2)
_, _, logS3 = decompose(D3)

plt.subplot(2, 2, 1) ; sns.heatmap(logS,  cbar=False)
plt.subplot(2, 2, 2) ; sns.heatmap(logS1, cbar=False)
plt.subplot(2, 2, 3) ; sns.heatmap(logS2, cbar=False)
plt.subplot(2, 2, 4) ; sns.heatmap(logS3, cbar=False)
plt.show()

wavfile.write('y_griffinlim_conj.wav', hp.sample_rate, y3)
