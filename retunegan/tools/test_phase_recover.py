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
    logS = np.log(S.clip(1e-5, None))
    return P, S, logS


dp = r'C:\Users\Kahsolt\Desktop\Workspace\Data\DataBaker\Wave'
fn = R.choice([fn for fn in os.listdir(dp) if fn.endswith('.wav')])
fp = os.path.join(dp, fn)

sr, hsr = hp.sample_rate, hp.sample_rate//2
y = L.load(fp, hp.sample_rate)[0]
if len(y) % 2 != 0: y = y[:-1]
ylen = len(y)


print('>> effect of noise in time domain')
D = L.stft(y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
P, S, logS = decompose(D)
for eps in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
    y_n = y + np.random.uniform(low=-eps, high=eps, size=y.shape)
    D_n = L.stft(y_n, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
    P_n, S_n, logS_n = decompose(D_n)
    print('eps =', eps)
    print('|y - y_n|:',       np.mean(np.abs(y - y_n)))
    print('|P - P_n|:',       np.mean(np.abs(P - P_n)))
    print('|S - S_n|:',       np.mean(np.abs(S - S_n)))
    print('|logS - logS_n|:', np.mean(np.abs(logS - logS_n)))
    print()
    if False: sns.heatmap(logS_n) ; plt.show()


print('>> reverse with original magnitude & original phase')
y_i = L.istft(D, win_length=hp.win_length, hop_length=hp.hop_length, length=ylen)
D_i = L.stft(y_i, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
P_i, S_i, logS_i = decompose(D_i)

print('|y - y_i|:',       np.mean(np.abs(y - y_i)))
print('|P - P_i|:',       np.mean(np.abs(P - P_i)))
print('|S - S_i|:',       np.mean(np.abs(S - S_i)))
print('|logS - logS_i|:', np.mean(np.abs(logS - logS_i)))
print()


print('>> reverse with original magnitude by GriffinLim')      # 这个步骤自带一定随机性
y_gl = L.griffinlim(S, hop_length=hp.hop_length, win_length=hp.win_length, length=ylen)
D_gl = L.stft(y_gl, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
P_gl, S_gl, logS_gl = decompose(D_gl)

print('|y - y_gl|:',       np.mean(np.abs(y - y_gl)))
print('|P - P_gl|:',       np.mean(np.abs(P - P_gl)))
print('|S - S_gl|:',       np.mean(np.abs(S - S_gl)))
print('|logS - logS_gl|:', np.mean(np.abs(logS - logS_gl)))
print()


print('>> reverse with original magnitude & random phase')
S_r, logS_r = S, logS
P_r = np.random.uniform(low=-np.pi, high=np.pi, size=P.shape)
D_r = S_r * np.exp(1j * P_r)
y_r = L.istft(D_r, win_length=hp.win_length, hop_length=hp.hop_length, length=ylen)
D_rr = L.stft(y_r, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
P_rr, S_rr, logS_rr = decompose(D_rr)

print('|y - y_r|:',          np.mean(np.abs(y - y_r)))
print('|P - P_r|:',          np.mean(np.abs(P - P_r)))
print('|P - P_rr|:',         np.mean(np.abs(P - P_rr)))
print('|P_r - P_rr|:',       np.mean(np.abs(P_r - P_rr)))
print('|S - S_rr|:',         np.mean(np.abs(S - S_rr)))
print('|logS - logS_rr|:',   np.mean(np.abs(logS - logS_rr)))
print()

print('>> reverse with random magnitude & original phase')
# 这些都不行
#S_f = np.exp(np.random.normal(loc=logS.mean(), scale=logS.std(), size=S.shape))/10
# S_f = np.random.normal(loc=S.mean(), scale=S.std(), size=S.shape)
# 这些可以
#S_f = np.random.normal(loc=logS.mean(), scale=logS.std(), size=S.shape)
#S_f = np.random.normal(loc=S.mean(), scale=S.mean(), size=S.shape)
S_f = np.random.uniform(low=S.min(), high=S.max(), size=S.shape)
P_f = P
D_f = S_f * np.exp(1j * P_f)
y_f = L.istft(D_f, win_length=hp.win_length, hop_length=hp.hop_length, length=ylen)
D_fr = L.stft(y_f, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
P_fr, S_fr, logS_fr = decompose(D_fr)

print('|y - y_f|:',          np.mean(np.abs(y - y_f)))
print('|P - P_fr|:',         np.mean(np.abs(P - P_fr)))
print('|S - S_f|:',          np.mean(np.abs(S - S_f)))
print('|S - S_fr|:',         np.mean(np.abs(S - S_fr)))
print()

#if True: sns.heatmap(S_f) ; plt.show()
#if True: sns.heatmap(P_fr) ; plt.show()
if True: sns.heatmap(S_fr) ; plt.show()
