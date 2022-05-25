#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/04/16 

import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

import hparam as hp
from audio import get_stft_torch
from utils import PI


# globals
MaxPool = nn.MaxPool1d(hp.envelope_pool_k)


# multi-stft loss
def multi_stft_loss(y, y_g, ret_loss=False, ret_specs=False):
  loss = 0
  if ret_specs: stft_r, stft_g = [], []

  # [B, 1, T] => [B, T]
  if len(y.shape) == 3:
    y, y_g = y.squeeze(1), y_g.squeeze(1)
  
  for n_fft, win_length, hop_length in hp.multi_stft_params:
    # 得到原始的Mel值
    y_mag,   y_mel,   y_phase   = get_stft_torch(y,   n_fft, win_length, hop_length)  # TODO: 这项可以拆出去缓存
    y_g_mag, y_g_mel, y_g_phase = get_stft_torch(y_g, n_fft, win_length, hop_length)

    # 得到对数放缩的谱值
    log_y_mel, log_y_g_mel = torch.log(y_mel), torch.log(y_g_mel)
    log_y_mag, log_y_g_mag = torch.log(y_mag), torch.log(y_g_mag)
    norm_y_phase, norm_y_g_phase = y_phase / PI, y_g_phase / PI

    if ret_specs:
      # 判别器在线性谱上作判定
      if hp.phd_input == 'stft':
        stft_r.append(torch.stack([log_y_mag,   norm_y_phase],   dim=1))
        stft_g.append(torch.stack([log_y_g_mag, norm_y_g_phase], dim=1))
      elif hp.phd_input == 'phase':
        stft_r.append(torch.stack([log_y_mag, norm_y_phase],   dim=1))
        stft_g.append(torch.stack([log_y_mag, norm_y_g_phase], dim=1))
      else: raise

    # 谱损失在Mel的对数值和原始值上作考察
    loss += F.l1_loss(    y_mel,     y_g_mel)
    loss += F.l1_loss(log_y_mel, log_y_g_mel)

  loss /= len(hp.multi_stft_params)
  
  if ret_loss and ret_specs:
    return loss, (stft_r, stft_g)
  elif ret_loss:
    return loss
  elif ret_specs:
    return (stft_r, stft_g)
  else: raise


# envelope loss for waveform
def envelope_loss(y, y_g):
  # 绝对动态包络
  loss = 0
  loss += torch.mean(torch.abs(MaxPool( y) - MaxPool( y_g)))
  loss += torch.mean(torch.abs(MaxPool(-y) - MaxPool(-y_g)))

  return loss


# dynamic loss for waveform
def dynamic_loss(y, y_g):
  # 相对动态大小
  dyn_y   = torch.abs(MaxPool(y)   + MaxPool(-y))
  dyn_y_g = torch.abs(MaxPool(y_g) + MaxPool(-y_g))
  loss    = torch.mean(torch.abs(dyn_y - dyn_y_g))

  return loss


# strip mirror loss for waveform
def strip_mirror_loss(y):
  # 可能没啥用甚至有副作用的正则损失

  # assure length is even
  if y.shape[-1] % 2 != 0: y = y[:,:,:-1]
  # strip split & de-mean
  even, odd = y[:,:,::2], y[:,:,1::2]
  even = even - even.mean()
  odd  = odd  - odd .mean()
  # maximize |e-o|
  loss = torch.mean(-torch.log(torch.clamp_max((torch.abs(even - odd) + 1e-9), max=1.0)))

  return loss


# adversarial loss for discriminator
def discriminator_loss(disc_r, disc_g):
  loss = 0

  for dr, dg in zip(disc_r, disc_g):    # List[[B, T], ...]
    if hp.relative_gan_loss:
      # maxmize gap betwwen dg & dr
      #r_loss = torch.mean((1 - (dr - dg.detach().mean())) ** 2)
      #r_loss = torch.mean((1 - (dr - dg.detach().mean(axis=-1))) ** 2)
      #r_loss = torch.mean(1 - torch.tanh(dr - dg.detach()))
      #r_loss = torch.mean((1 - (dr - dg.detach())) ** 2)
      #g_loss = torch.mean((0 - dg) ** 2)
      r_loss = torch.mean(torch.mean((1 - (dr - dg.detach())) ** 2, axis=-1))
      g_loss = torch.mean(torch.mean((0 - dg) ** 2, axis=-1))
      #r_loss = torch.mean(-torch.log(dr))
      #g_loss = torch.mean(dg)
    else:
      # let dr -> 1, dg -> 0
      #r_loss = torch.mean((1 - dr) ** 2)
      #g_loss = torch.mean((0 - dg) ** 2)
      r_loss = torch.mean(torch.mean((1 - dr) ** 2, axis=-1))
      g_loss = torch.mean(torch.mean((0 - dg) ** 2, axis=-1))
    loss += (r_loss + g_loss)

  return loss


# adversarial loss for generator
def generator_loss(disc_g, disc_r):
  loss = 0

  for dg, dr in zip(disc_g, disc_r):
    if hp.relative_gan_loss:
      # let dg ~= dr
      #g_loss = torch.mean((dr.detach().mean(axis=-1) - dg) ** 2)
      #g_loss = torch.mean(1 - torch.tanh(dg - dr.detach()))
      #g_loss = torch.mean((dr.detach() - dg) ** 2)
      g_loss = torch.mean(torch.mean((dg - dr.detach()) ** 2, axis=-1))
    else:
      # let dg -> 1
      #g_loss = torch.mean((1 - dg) ** 2)
      g_loss = torch.mean(torch.mean((1 - dg) ** 2, axis=-1))
    loss += g_loss

  return loss


# feature map loss for generator
def feature_loss(fmap_r, fmap_g):
  loss = 0

  for dr, dg in zip(fmap_r, fmap_g):
    for r, g in zip(dr, dg):
      loss += F.l1_loss(r, g)

  return loss
