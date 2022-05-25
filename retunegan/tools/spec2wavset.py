#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/02/14 

# 借鉴RefineGAN的想法:
#   RefineGAN: 根据从上游模型预测生成的f0,c0,voice-flag信息，生成一个wavform template
#              对于unvoiced段填充高斯白噪、对于voice段依据f0的频率来放置脉冲，再依照c0画整体包络
#   spec2wavset: stft谱展示了将原信号拆解为一组频率等间距的正弦信号，因此我们可以直接叠加这一组正弦波作为wavform template
#                Q: 为什么不用GriffinLim的输出做模板呢，后续网络相当于在频域进行降噪了
#                A: 降噪是很困难的，但是逆向思考——正弦波的组合是比较干净的、我们基于它来加噪声!
#     *注意: 需要多组stft参数来避免频率丢失、减缓窗函数导致的频率泄露问题
#      fft_params:
#            n_fft    win_length    hop_length
#            2048       1024        256
#            1024       512         128
#             512       256          64

# 从FFT提取频率和振幅: https://blog.csdn.net/taw19960426/article/details/101684663
# 宽带语谱图: 帧长 3ms, hop_length=48(16000Hz)/66(22050Hz)
# 窄带语谱图: 帧长20ms, hop_length=320(16000Hz)/441(22050Hz)

# NOTE: 统计特征
#   magnitude: 单峰(语音信号) + 左侧高原(底噪)
#   phase: 近乎[-pi,pi]间的均匀分布

import os
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt
import seaborn as sns
import librosa as L
from scipy.io import wavfile

sample_rate = 16000
n_mel       = 80
fmin        = 70
fmax        = 7600
fft_params  = [
  # (n_fft, win_length, hop_length)
  (4096, 2048,  512),
  (2048, 1024,  256),
  (1024,  512,  128),
  ( 512,  256,   64),
  ( 256,  128,   32),
]


def calc_spec(y, n_fft, hop_length, win_length, clamp_lower=False):
  D = L.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')
  S = np.abs(D)
  print('S.min():', S.min())
  mag = np.log(S.clip(min=1e-5)) if clamp_lower else S
  M = L.feature.melspectrogram(S=S, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                               n_mels=n_mel, fmin=fmin, fmax=fmax, window='hann', power=1, htk=True)
  print('M.min():', M.min())
  mel = np.log(M.clip(min=1e-5)) if clamp_lower else M
  return mag, mel


def get_specs(fp):
  y, _ = L.load(fp, sr=sample_rate, mono=True, res_type='kaiser_best')
  return [calc_spec(y, n_fft, hop_length, win_length) for n_fft, win_length, hop_length in fft_params]


def display_specs(specs):
  n_fig = len(specs)
  for i in range(n_fig):
    if isinstance(type(specs[i]), (list, tuple)):
      plt.subplot(n_fig*100+10+i+1)
      ax = sns.heatmap(specs[i][0])
      ax.invert_yaxis()
      plt.subplot(n_fig*100+20+i+1)
      ax = sns.heatmap(specs[i][1])
      ax.invert_yaxis()
    else:
      plt.subplot(n_fig*100+10+i+1)
      ax = sns.heatmap(specs[i])
      ax.invert_yaxis()
  plt.show()
  plt.clf()

def f(k, n_fft):
  return k * (sample_rate / n_fft)

def sin(x, A, freq, phi=0):
  w = 2 * np.pi * freq      # f = w / (2*pi)
  return A * np.sin(w * x + phi)


def extract_f_A(fp):
  for i, (mag, _) in enumerate(get_specs(fp)):
    n_fft, _, _ = fft_params[i]
    print(f'[n_fft={n_fft}]')

    fr = sample_rate / n_fft    # 频率分辨率单位
    print('mag min/max/mean:', mag.min(), mag.max(), mag.mean())

    f_A = []
    thresh = mag.mean() * 2
    for i, energy in enumerate(mag):
      print(f'frame {i}: ')
      j = 0
      while j < len(energy):
        # 阈值响应
        if energy[j] <= thresh:
          j += 1
        else:
          # 略过上坡
          while j + 1 < len(energy) and energy[j+1] >= energy[j]:
            j += 1
          # 取得峰值
          f_A.append((fr * j, energy[j]))
          print(f'   freq({j}) = {fr * j} Hz, amp = {energy[j]}')
          # 略过下坡
          while j + 1 < len(energy) and energy[j+1] < energy[j]:
            j += 1


def demo_extract_f_A():
  st = 1    # 采样时间 1s
  x = np.arange(0, 1, st / sample_rate)
  print('signal len:', len(x))
  y1 = sin(x, 2, 207)
  y2 = sin(x, -1, 843)
  y = y1 + y2

  plt.subplot(311) ; plt.plot(x, y1)
  plt.subplot(312) ; plt.plot(x, y2)
  plt.subplot(313) ; plt.plot(x, y)
  plt.show()

  n_fft = 2048
  mag = calc_mag(y, n_fft=n_fft, win_length=1024, hop_length=256)
  display_specs(mag)

  mag = mag.T
  
  n_fft = fft_params[1][0]
  A = 2 * mag / n_fft
  fr = sample_rate / n_fft    # 频率分辨率单位

  f_A = []
  thresh = 0.1
  for i, energy in enumerate(A):
    print(f'frame {i}: ')
    j = 0
    while j < len(energy):
      # 阈值响应
      if energy[j] <= thresh:
        j += 1
      else:
        # 略过上坡
        while j + 1 < len(energy) and energy[j+1] >= energy[j]:
          j += 1
        # 取得峰值
        f_A.append((fr * j, energy[j]))
        print(f'   freq({j}) = {fr * j} Hz, amp = {energy[j]}')
        # 略过下坡
        while j + 1 < len(energy) and energy[j+1] < energy[j]:
          j += 1

  return f_A


def demo_stft_istft(fp):
  n_fft, win_length, hop_length = 2048, 1024, 256

  y, _ = L.load(fp, sr=sample_rate, mono=True, res_type='kaiser_best')
  length = len(y)   # for align output

  stft = L.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
  mag, phase = np.abs(stft), np.angle(stft)
  y_istft = L.istft(stft, win_length=win_length, hop_length=hop_length, length=length)
  D = L.stft(y_istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
  mag_istft, phase_istft = np.abs(D), np.angle(D)
  
  print('mag.min():', mag.min(), 'mag.max():', mag.max())
  plt.subplot(311) ; plt.hist(mag.flatten(), bins=150)            # [0, 10]
  maglog = np.log(mag)
  print('maglog.min():', maglog.min(), 'maglog.max():', maglog.max())
  plt.subplot(312) ; plt.hist(maglog.flatten(), bins=150)         # [-12, 3]
  plt.subplot(313) ; plt.hist(phase.flatten(), bins=150)          # [-pi, pi]
  plt.show()

  phase_ristft_r = np.random.uniform(low=-np.pi, high=np.pi, size=phase.shape)
  rstft = mag * np.exp(1j * phase_ristft_r)
  y_ristft = L.istft(rstft, win_length=win_length, hop_length=hop_length, length=length)
  D = L.stft(y_ristft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
  mag_ristft, phase_ristft = np.abs(D), np.angle(D)

  y_gl = L.griffinlim(mag, hop_length=hop_length, win_length=win_length, length=length)
  D = L.stft(y_gl, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
  mag_gl, phase_gl = np.abs(D), np.angle(D)

  print('y_istft error:',      np.sum(y - y_istft))
  print('y_ristft error:',     np.sum(y - y_ristft))
  print('y_gl error:',         np.sum(y - y_gl))

  print('y_istft mirror error:',  np.sum([e - o for e, o in zip(y_istft[::2], y_istft[1::2])]))
  print('y_ristft mirror error:', np.sum([e - o for e, o in zip(y_ristft[::2], y_ristft[1::2])]))
  print('y_gl mirror error:',     np.sum([e - o for e, o in zip(y_gl[::2], y_gl[1::2])]))

  print('mag_istft error:',    np.sum(mag - mag_istft))
  print('mag_ristft error:',   np.sum(mag - mag_ristft))
  print('mag_gl error:',       np.sum(mag - mag_gl))

  print('phase_istft error:',    np.sum(phase - phase_istft))
  print('phase_ristft error:',   np.sum(phase - phase_ristft))
  print('phase_ristft_r error:', np.sum(phase_ristft_r - phase_ristft))
  print('phase_gl error:',       np.sum(phase - phase_gl))

  plt.subplot(231) ; sns.heatmap(mag)
  plt.subplot(232) ; sns.heatmap(mag_ristft)
  plt.subplot(233) ; sns.heatmap(mag_gl)
  plt.subplot(234) ; sns.heatmap(phase)
  plt.subplot(235) ; sns.heatmap(phase_ristft)
  plt.subplot(236) ; sns.heatmap(phase_gl)
  plt.show()

  wavfile.write('y.wav',        sample_rate, y)
  wavfile.write('y_istft.wav',  sample_rate, y_istft)
  wavfile.write('y_ristft.wav', sample_rate, y_ristft)
  wavfile.write('y_gl.wav',     sample_rate, y_gl)


if __name__ == '__main__':
  dp = r'D:\Desktop\Workspace\Data\DataBaker\Wave'
  fn = R.choice([fn for fn in os.listdir(dp) if fn.endswith('.wav')])
  fp = os.path.join(dp, fn)

  demo_stft_istft(fp)
  #extract_f_A(fp)
