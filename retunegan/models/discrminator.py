#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/04/16 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

import hparam as hp
from utils import *

PI = 3.14159265358979


class DiscriminatorS(nn.Module):

  def __init__(self, use_sn=False):
    super().__init__()
    
    #norm_f = spectral_norm if use_sn else weight_norm 

    # [8192]; 降采样 4*4*4*4=256 倍，对于各个降采样版本即降采样[256,512,1024]倍
    sel = 'MelGAN_small'
    if sel == 'MelGAN':
      self.convs = nn.ModuleList([
        weight_norm(Conv1d(   1,   16, 15, 1, padding=7)),
        weight_norm(Conv1d(  16,   64, 41, 4, padding=20, groups=4 )),
        weight_norm(Conv1d(  64,  256, 41, 4, padding=20, groups=16)),
        weight_norm(Conv1d( 256, 1024, 41, 4, padding=20, groups=64)),
        weight_norm(Conv1d(1024, 1024, 41, 4, padding=20, groups=256)),
        weight_norm(Conv1d(1024, 1024,  5, 1, padding=2)),
      ])
      self.conv_post = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))
    if sel == 'MelGAN_small':
      self.convs = nn.ModuleList([
        weight_norm(Conv1d(  1,  32, 15, 1, padding=7)),
        weight_norm(Conv1d( 32,  64, 41, 2, padding=20, groups=4 )),
        weight_norm(Conv1d( 64, 128, 41, 2, padding=20, groups=8 )),
        weight_norm(Conv1d(128, 512, 41, 4, padding=20, groups=32)),
        weight_norm(Conv1d(512, 512, 41, 4, padding=20, groups=64)),
        weight_norm(Conv1d(512, 512,  5, 1, padding=2)),
      ])
      self.conv_post = weight_norm(Conv1d(512, 1, 3, 1, padding=1))
    elif sel == 'HiFiGAN':
      self.convs = nn.ModuleList([
        weight_norm(Conv1d(   1,  128, 15, 1, padding=7)),
        weight_norm(Conv1d( 128,  128, 41, 2, padding=20, groups=4 )),
        weight_norm(Conv1d( 128,  256, 41, 2, padding=20, groups=16)),
        weight_norm(Conv1d( 256,  512, 41, 4, padding=20, groups=16)),
        weight_norm(Conv1d( 512, 1024, 41, 4, padding=20, groups=16)),
        weight_norm(Conv1d(1024, 1024, 41, 1, padding=20, groups=16)),
        weight_norm(Conv1d(1024, 1024,  5, 1, padding=2)),
      ])
      self.conv_post = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))

  def forward(self, x):
    # torch.Size([16, 1, 8192])
    # torch.Size([16, 32, 8192])
    # torch.Size([16, 64, 4096])
    # torch.Size([16, 128, 2048])
    # torch.Size([16, 512, 512])
    # torch.Size([16, 512, 128])
    # torch.Size([16, 512, 128])
    # torch.Size([16, 1, 128])
    #
    # torch.Size([16, 1, 4096])
    # torch.Size([16, 32, 4096])
    # torch.Size([16, 64, 2048])
    # torch.Size([16, 128, 1024])
    # torch.Size([16, 512, 256])
    # torch.Size([16, 512, 64])
    # torch.Size([16, 512, 64])
    # torch.Size([16, 1, 64])
    #
    # torch.Size([16, 1, 2048])
    # torch.Size([16, 32, 2048])
    # torch.Size([16, 64, 1024])
    # torch.Size([16, 128, 512])
    # torch.Size([16, 512, 128])
    # torch.Size([16, 512, 32])
    # torch.Size([16, 512, 32])
    # torch.Size([16, 1, 32])

    DEBUG = 0
    if DEBUG: print('[DiscriminatorS]')

    fmap = []
    if DEBUG: print(x.shape)

    for i, l in enumerate(self.convs):
      x = l(x)
      if DEBUG: print(x.shape)
      fmap.append(x)
      x = F.leaky_relu(x, LRELU_SLOPE)
    x = self.conv_post(x)
    if DEBUG: print(x.shape)
    x = torch.flatten(x, 1, -1)

    return x, fmap


class MultiScaleDiscriminator(nn.Module):

  def __init__(self):
    super().__init__()

    self.discriminators = nn.ModuleList([
      DiscriminatorS(use_sn=i==0) for i in range(hp.msd_layers)
    ])
    # 不能使用音频处理的Resample，而要用AvgPool逐步抹去高频细节
    self.avgpool = nn.AvgPool1d(kernel_size=hp.downsample_pool_k, stride=2, padding=1)

  def forward(self, y, y_hat):
    y_d_rs,  y_d_gs  = [], []
    fmap_rs, fmap_gs = [], []

    for i, d in enumerate(self.discriminators):
      y_d_r, fmap_r = d(y)
      y_d_g, fmap_g = d(y_hat)
      y_d_rs.append(y_d_r) ; fmap_rs.append(fmap_r)
      y_d_gs.append(y_d_g) ; fmap_gs.append(fmap_g)

      if i != len(self.discriminators) - 1:
        y     = self.avgpool(y)
        y_hat = self.avgpool(y_hat)
      
    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):

  def __init__(self, period):
    super().__init__()

    self.period = period
    
    # [8192]; 只在T方向降采样 3*3*3*3=81 倍
    #   = [4096，2]
    #   = [2731，3] (*)
    #   = [1639，5] (*)
    #   = [1171，7] (*)
    #   = [745，11] (*)
    sel = 'HiFiGAN_small'
    if sel == 'HiFiGAN':
      self.convs = nn.ModuleList([
        weight_norm(Conv2d(   1,   32, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d(  32,  128, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d( 128,  512, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d( 512, 1024, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d(1024, 1024, (5, 1), 1,      padding=(2, 0))),
      ])
      self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    elif sel == 'HiFiGAN_small':
      self.convs = nn.ModuleList([
        weight_norm(Conv2d(  1,  32, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d( 32, 128, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d(128, 256, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d(256, 512, (5, 1), (3, 1), padding=(2, 0))),
        weight_norm(Conv2d(512, 512, (5, 1), 1,      padding=(2, 0))),
      ])
      self.conv_post = weight_norm(Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

  def forward(self, x):
    # torch.Size([16, 1, 2731, 3])
    # torch.Size([16, 32, 911, 3])
    # torch.Size([16, 128, 304, 3])
    # torch.Size([16, 256, 102, 3])
    # torch.Size([16, 512, 34, 3])
    # torch.Size([16, 512, 34, 3])
    # torch.Size([16, 1, 34, 3])
    #
    # torch.Size([16, 1, 1639, 5])
    # torch.Size([16, 32, 547, 5])
    # torch.Size([16, 128, 183, 5])
    # torch.Size([16, 256, 61, 5])
    # torch.Size([16, 512, 21, 5])
    # torch.Size([16, 512, 21, 5])
    # torch.Size([16, 1, 21, 5])
    #
    # torch.Size([16, 1, 1171, 7])
    # torch.Size([16, 32, 391, 7])
    # torch.Size([16, 128, 131, 7])
    # torch.Size([16, 256, 44, 7])
    # torch.Size([16, 512, 15, 7])
    # torch.Size([16, 512, 15, 7])
    # torch.Size([16, 1, 15, 7])
    #
    # torch.Size([16, 1, 745, 11])
    # torch.Size([16, 32, 249, 11])
    # torch.Size([16, 128, 83, 11])
    # torch.Size([16, 256, 28, 11])
    # torch.Size([16, 512, 10, 11])
    # torch.Size([16, 512, 10, 11])
    # torch.Size([16, 1, 10, 11])

    DEBUG = 0
    if DEBUG: print('[DiscriminatorP]')

    fmap = []

    # 1d to 2d
    b, c, t = x.shape
    if t % self.period != 0:  # pad tail
      n_pad = self.period - (t % self.period)
      x = F.pad(x, (0, n_pad), "reflect")
      t = t + n_pad
    # [B, C, T', P]
    x = x.view(b, c, t // self.period, self.period)

    if DEBUG: print(x.shape)
    for i, l in enumerate(self.convs):
      x = l(x)
      if DEBUG: print(x.shape)
      fmap.append(x)
      x = F.leaky_relu(x, LRELU_SLOPE)
    x = self.conv_post(x)
    if DEBUG: print(x.shape)
    x = torch.flatten(x, 1, -1)

    return x, fmap


class MultiPeriodDiscriminator(nn.Module):

  def __init__(self):
    super().__init__()

    self.discriminators = nn.ModuleList([
      DiscriminatorP(hp.mpd_periods[i]) for i in range(len(hp.mpd_periods))
    ])

  def forward(self, y, y_hat):
    y_d_rs,  y_d_gs  = [], []
    fmap_rs, fmap_gs = [], []
    
    for d in self.discriminators:
      y_d_r, fmap_r = d(y)
      y_d_g, fmap_g = d(y_hat)
      y_d_rs.append(y_d_r) ; fmap_rs.append(fmap_r)
      y_d_gs.append(y_d_g) ; fmap_gs.append(fmap_g)

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class StftDiscriminator(nn.Module):

  def __init__(self, i, ch=2):
    super().__init__()

    # [1025, 35]
    # [513,  69]
    # [257, 137]
    self.convs = nn.ModuleList([
      weight_norm(Conv2d( ch,  32, (3, 3), (2, 1), padding=(1, 1))),
      weight_norm(Conv2d( 32,  64, (3, 3), (2, 2), padding=(1, 1))),
      weight_norm(Conv2d( 64, 256, (5, 3), (3, 2), padding=(2, 1))),
      weight_norm(Conv2d(256, 512, (5, 3), (3, 2), padding=(2, 1))),
      weight_norm(Conv2d(512, 512, 3, 1, padding=1)),
    ])
    self.conv_post = weight_norm(Conv2d(512, 1, 3, 1, padding=1))

    self.convs  .apply(init_weights)
    self.conv_post.apply(init_weights)

  def forward(self, x):
    # torch.Size([16, 2, 1025, 35])
    # torch.Size([16, 32, 513, 35])
    # torch.Size([16, 64, 257, 18])
    # torch.Size([16, 256, 86, 9])
    # torch.Size([16, 512, 29, 5])
    # torch.Size([16, 512, 29, 5])
    # torch.Size([16, 1, 29, 5])
    # 
    # torch.Size([16, 2, 513, 69])
    # torch.Size([16, 32, 257, 69])
    # torch.Size([16, 64, 129, 35])
    # torch.Size([16, 256, 43, 18])
    # torch.Size([16, 512, 15, 9])
    # torch.Size([16, 512, 15, 9])
    # torch.Size([16, 1, 15, 9])
    # 
    # torch.Size([16, 2, 257, 137])
    # torch.Size([16, 32, 129, 137])
    # torch.Size([16, 64, 65, 69])
    # torch.Size([16, 256, 22, 35])
    # torch.Size([16, 512, 8, 18])
    # torch.Size([16, 512, 8, 18])
    # torch.Size([16, 1, 8, 18])

    DEBUG = 0
    if DEBUG: print('[StftDiscriminator]')

    fmap = []
    if DEBUG: print(x.shape)

    # x.shape = [B, 2, C, T]
    for i, l in enumerate(self.convs):
      x = l(x)
      if DEBUG: print(x.shape)
      fmap.append(x)
      x = F.leaky_relu(x, LRELU_SLOPE)
    x = self.conv_post(x)
    if DEBUG: print(x.shape)
    x = torch.flatten(x, 1, -1)

    return x, fmap


class MultiStftDiscriminator(nn.Module):

  def __init__(self):
    super().__init__()

    self.discriminators = nn.ModuleList([
      StftDiscriminator(i) for i in range(len(hp.multi_stft_params))
    ])

  def forward(self, phs, ph_hats):
    ph_d_rs, ph_d_gs = [], []
    fmap_rs, fmap_gs = [], []

    for d, ph, ph_hat in zip(self.discriminators, phs, ph_hats):
      ph_d_r, fmap_r = d(ph)
      ph_d_g, fmap_g = d(ph_hat)
      ph_d_rs.append(ph_d_r) ; fmap_rs.append(fmap_r)
      ph_d_gs.append(ph_d_g) ; fmap_gs.append(fmap_g)
      
    return ph_d_rs, ph_d_gs, fmap_rs, fmap_gs
