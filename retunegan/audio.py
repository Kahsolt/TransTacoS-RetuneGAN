#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/01/07 

import torch
from torch.nn import AvgPool2d
import numpy as np
import numpy.random as R
import librosa as L
from scipy.io import wavfile

import seaborn as sns
import matplotlib.pyplot as plt

import hparam as hp
R.seed(hp.randseed)


eps = 1e-5
mel_basis  = L.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.n_mel, fmin=hp.fmin, fmax=hp.fmax)
mag_to_mel = lambda x: np.dot(mel_basis, x)

avg_pool_2d = AvgPool2d(kernel_size=3, stride=1, padding=1)

mel_basis_torch = { }   # { n_fft: mel_basis }
window_fn_torch = { }   # { win_length: window_fn }


def load_wav(path):  # float values in range (-1,1)
  y, _ = L.load(path, sr=hp.sample_rate, mono=True, res_type='kaiser_best')
  return y.astype(np.float32)     # [T,]


def save_wav(wav, path):
  wavfile.write(path, hp.sample_rate, wav)


def align_wav(wav, r=hp.hop_length):
  d = len(wav) % r
  if d != 0:
    wav = np.pad(wav, (0, (r - d)))
  return wav


def augment_wav(y, pitch_shift=True, time_stretch=True, dynamic_scale=True):
  if pitch_shift:
    # 75% unmodified, 25% shifted
    if R.random() > 0.75:
      # ~10% unmodified, ~30% in (-1,1), ~47% in (-2,+2), ~%74 in (-4,+4)
      semitone = max(min(round(R.normal(scale=12/3)), 12), -12)   # 3-sigma principle:99.74% in [mu-3*sigma,mu+3*sigma]
      if semitone != 0: y = L.effects.pitch_shift(y, hp.sample_rate, semitone, res_type='kaiser_best')
  
  if time_stretch:
    # 90% unmodified, 10% twisted; because `time_stretch`` hurts quality a lot
    if R.random() > 0.90:
      alpha = 2 ** R.normal(scale=1/5)
      if abs(alpha - 1.0) < 0.1: alpha = 1.0
      if alpha != 1.0: y = L.effects.time_stretch(y=y, rate=alpha, win_length=hp.win_length, hop_length=hp.hop_length)

  if dynamic_scale:
    # 25% unmodified, 75% global shift
    r = R.random()
    if r > 0.25:
      alpha = 2 ** R.normal(scale=1/3)
      y = y * alpha
      absmax = max(y.max(), -y.min())
      if absmax > 1.0: y /= absmax

  return y.astype(np.float32)     # [T,]


def augment_spec(S, time_mask=True, freq_mask=True, prob=0.2, rounds=3, freq_width=9, time_width=3):
  F, T = S.shape
  S = torch.from_numpy(S).unsqueeze(0)

  # local mask
  # 10.7% unmodified, 57.0% maskes <=2 times, 86.0% masked <=4 times 
  for _ in range(rounds):
    if freq_mask and R.random() < prob:
      s = R.randint(0, F - freq_width)
      r = R.randint(1, freq_width)
      mask_val = R.uniform(low=S.min(), high=S.mean())
      S[:, s:s+r, :] = torch.ones([1, r, T]) * mask_val

    if time_mask and R.random() < prob:
      s = R.randint(0, T - time_width)
      r = R.randint(1, time_width)
      mask_val = R.uniform(low=S.min(), high=S.mean())
      S[:, :, s:s+r] = torch.ones([1, F, r]) * mask_val
  
  # global blur
  S = avg_pool_2d(S)

  S = S.squeeze(0).numpy()
  return S.astype(np.float32)


def get_zcr(y):
  zcr = L.feature.zero_crossing_rate(y, frame_length=hp.win_length, hop_length=hp.hop_length)[0]
  return zcr.astype(np.float32)     # [T,]


def get_c0(y):
  c0 = L.feature.rms(y=y, frame_length=hp.win_length, hop_length=hp.hop_length)[0]
  return c0.astype(np.float32)       # [T,]


def get_uv(zcr, dyn):
  uv = np.empty_like(zcr)
  for i in range(len(uv)):
    # NOTE: these numbers are magic, tune by hand according to your dataset
    uv[i] = zcr[i] > 0.18 or dyn[i] < 0.03
  return uv


def get_mag(y, clamp_low=True):
  D = L.stft(y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn)
  S = np.abs(D)
  mag = np.log(S.clip(min=eps) if clamp_low else S)
  return mag.astype(np.float32)      # [F, T]


def get_mel(y, clamp_low=True):
  M = L.feature.melspectrogram(y=y, sr=hp.sample_rate, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, 
                                n_mels=hp.n_mel, fmin=hp.fmin, fmax=hp.fmax,
                                window=hp.window_fn, power=1, htk=hp.mel_scale=='htk')
  mel = np.log(M.clip(min=eps) if clamp_low else M)
  return mel.astype(np.float32)      # [M, T]


def _griffinlim(S, wavlen=None):
  if hp.gl_power: S = S ** hp.gl_power
  y = L.griffinlim(S, n_iter=hp.gl_iters, 
                   hop_length=hp.hop_length, win_length=hp.win_length, window=hp.window_fn, 
                   length=wavlen, momentum=hp.gl_momentum, init='random', random_state=hp.randseed)
  return y.astype(np.float32)


def inv_mag(mag, wavlen=None):
  S = np.exp(mag)   # [F/F-1, T], reverse np.log
  F, T = mag.shape
  if F == hp.n_freq - 1:    # NOTE: preprend zero DC component
    S = np.concatenate([np.zeros([1, T]), S], axis=0)
  #print('S.min():', S.min(), 'S.max(): ', S.max(), 'S.mean(): ', S.mean())
  y = _griffinlim(S, wavlen)
  if wavlen: assert len(y) == wavlen
  return y


def get_stft_torch(y, n_fft, win_length, hop_length):
  ''' 该函数得到原始的Mel值，没有数值下截断和取对数过程 '''

  global mel_basis_torch, window_fn_torch
  if win_length not in window_fn_torch:
    win_functor = getattr(torch, f'{hp.window_fn}_window')
    window_fn_torch[win_length] = win_functor(win_length).to(y.device)            # [n_fft]
  if n_fft not in mel_basis_torch:
    mel_filter = L.filters.mel(hp.sample_rate, n_fft, hp.n_mel, hp.fmin, hp.fmax)
    mel_basis_torch[n_fft] = torch.from_numpy(mel_filter).float().to(y.device)    # [n_mel, n_fft//2+1]

  D = torch.stft(y, n_fft, return_complex=True,          # [n_fft/2+1, n_frames, 2], last dim 2 for real/image parts
                    hop_length=hop_length, win_length=win_length, window=window_fn_torch[win_length],   
                    center=True, pad_mode='reflect', normalized=False, onesided=True)
  
  #S = torch.sqrt(D.pow(2).sum(-1) + (1e-9))               # [n_fft/2+1, n_frames], get modulo, aka. magnitude
  S = torch.abs(D + 1e-9)
  M = torch.matmul(mel_basis_torch[n_fft], S)              # [n_mel, n_frames]
  P = torch.angle(D)

  return S, M, P
