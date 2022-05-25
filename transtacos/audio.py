#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/01/07 

import numpy as np
import librosa as L
from scipy import signal
from scipy.io import wavfile

import hparam as hp


eps      = 1e-5
firwin   = signal.firwin(hp.n_freq, [hp.fmin, hp.fmax], pass_zero=False, fs=hp.sample_rate)
rf0min   = L.note_to_hz(hp.rf0min) if isinstance(hp.rf0min, str) else float(hp.rf0min)
rf0max   = L.note_to_hz(hp.rf0max) if isinstance(hp.rf0max, str) else float(hp.rf0max)
c0min    = hp.c0min
c0max    = hp.c0max
qt_f0min = int(np.floor(L.hz_to_midi(hp.f0min)))
qt_f0max = int(np.ceil (L.hz_to_midi(hp.f0max)))

hp.n_f0_min  = qt_f0min
hp.n_f0_bins = qt_f0max - qt_f0min + 1

print(f'c0: min={c0min} max={c0max} n_bins={hp.n_c0_bins}')
print(f'qt_f0: min={qt_f0min} max={qt_f0max} n_bins={hp.n_f0_bins}')


def load_wav(path):  # float values in range (-1,1)
  y, _ = L.load(path, sr=hp.sample_rate, mono=True, res_type='kaiser_best')
  return y.astype(np.float32)     # [T,]

def save_wav(wav, path):
  if hp.postprocess:
    # rescaling for unified measure for all clips
    # NOTE: normalize amplification
    wav = wav / np.abs(wav).max() * 0.999
    # factor 0.5 in case of overflow for int16
    f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
    # sublinear scaling as Y ~ X ^ k (k < 1)
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.667)
    wav = f1 * f2

    # bandpass for less noises
    wav = signal.convolve(wav, firwin)

    wavfile.write(path, hp.sample_rate, wav.astype(np.int16))
  else:
    wavfile.write(path, hp.sample_rate, wav.astype(np.float32))


def align_wav(wav, r=hp.hop_length):
  d = len(wav) % r
  if d != 0:
    wav = np.pad(wav, (0, (r - d)))
  return wav


def trim_silence(wav, frame_length=512, hop_length=128):
  # 人声动态一般高达55dB
  return L.effects.trim(wav, top_db=hp.trim_below_peak_db, frame_length=frame_length, hop_length=hop_length)[0]


def preemphasis(x):       # 增强了高频, 听起来有点远场
  # x[i] = x[i] - k * x[i-1], k ie. preemphasis
  return signal.lfilter([1, -hp.preemphasis], [1], x)


def inv_preemphasis(x):   # undo preemphasis, should be put after Griffin-Lim
  return signal.lfilter([1], [1, -hp.preemphasis], x)


def get_specs(y):
  D = np.abs(_stft(preemphasis(y)))
  S = _amp_to_db(D) - hp.ref_level_db
  M = _amp_to_db(_linear_to_mel(D)) - hp.ref_level_db
  return (_normalize(S), _normalize(M))


def spec_to_natural_scale(spec):
  '''from inner normalized scale to raw scale of stft output'''
  return _db_to_amp(_denormalize(spec) + hp.ref_level_db)


def fix_zero_DC(S):
  F, T = S.shape
  if F == hp.n_freq - 1:    # NOTE: preprend zero DC component
    S = np.concatenate([np.ones([1, T]) * S.min() * 1e-2, S], axis=0)
    #S = np.concatenate([np.zeros([1, T]), S], axis=0)
  return S


def inv_spec(spec):
  S = spec_to_natural_scale(spec)                       # denorm
  S = fix_zero_DC(S)
  wav = inv_preemphasis(_griffin_lim(S ** hp.gl_power)) # reconstruct phase
  return wav.astype(np.float32)


def inv_mel(mel):     # This might have no use case
  M = spec_to_natural_scale(mel)                        # denorm
  S = _mel_to_linear(M)                                 # back to linear
  wav = inv_preemphasis(_griffin_lim(S ** hp.gl_power)) # reconstruct phase
  return wav.astype(np.float32)


def get_f0(y):
  f0 = L.yin(y, fmin=rf0min, fmax=rf0max, frame_length=hp.win_length, hop_length=hp.hop_length)
  return f0.astype(np.float32)       # [T,]


def get_c0(y):
  c0 = L.feature.rms(y=y, frame_length=hp.win_length, hop_length=hp.hop_length)[0]
  return c0.astype(np.float32)       # [T,]


def quantilize_f0(f0):
  f0 = np.asarray([L.hz_to_midi(f) - hp.n_f0_min for f in f0])
  f0 = f0.clip(0, hp.n_f0_bins - 1)
  return f0.astype(np.int32)         # [T,]


def quantilize_c0(c0):
  c0 = (c0 - c0min) / (c0max - c0min)
  c0 = c0 * hp.n_c0_bins
  c0 = c0.clip(0, hp.n_c0_bins - 1)
  return c0.astype(np.int32)         # [T,]


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hp.gl_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _stft(y):
  return L.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)


def _istft(y):
  return L.istft(y, hop_length=hp.hop_length, win_length=hp.win_length)


_mel_basis    = None
_linear_basis = None

def _linear_to_mel(spec):
  return np.dot(_get_mel_basis(), spec)

def _get_mel_basis():
  global _mel_basis
  if _mel_basis is None:
    assert hp.fmax < hp.sample_rate // 2
    _mel_basis = L.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.n_mel, fmin=hp.fmin, fmax=hp.fmax)
  return _mel_basis

def _mel_to_linear(mel):
  return np.dot(_get_linear_basis(), mel)

def _get_linear_basis():
  global _linear_basis
  if _linear_basis is None:
    m = _get_mel_basis()
    m_T = np.transpose(m)
    p = np.matmul(m, m_T)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    _linear_basis = np.matmul(m_T, np.diag(d))
  return _linear_basis

def _amp_to_db(x):
  # 人耳可听的声压范围为2e-5~20Pa，对应的声压级范围为0~120dB
  # 声压级公式 SPL = 20 * log10(p_e/p_ref)
  #   其中p_e为声压/振幅，参考声压p_ref为人耳最低可听觉声压、空气中一般取2e-5
  #   即有 SPL = 20 * log10(p_e/2e-5)
  #            = 20 * (log10(p_e) - log10(2e-5))
  #            ~= 20 * log10(p_e) - 94
  return 20 * np.log10(np.maximum(1e-5, x))    # 下截断、可有可无，只是作heatmap时方便查看

def _db_to_amp(x):
  # =10^(x/20)
  return np.power(10.0, x * 0.05)

def _normalize(S):
  # mapping [hp.min_level_db, 0] => [-hp.max_abs_value, hp.max_abs_value]
  # typically:         [-100, 0] => [-4, 4]
  return 2 * hp.max_abs_value * ((S - hp.min_level_db) / -hp.min_level_db) - hp.max_abs_value

def _denormalize(S):
  return ((S + hp.max_abs_value) * -hp.min_level_db) / (2 * hp.max_abs_value) + hp.min_level_db
