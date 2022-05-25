#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/04/16 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import hparam as hp
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''layers'''

class GaussianNoise(nn.Module):

  def __init__(self):
    super().__init__()

    self.w = nn.Parameter(torch.FloatTensor([1e-6]), requires_grad=True)
  
  def forward(self, x):
      n = torch.rand_like(x)
      x = x + n * self.w
      x = F.leaky_relu(x, LRELU_SLOPE)
      return x

## All about MelGAN ##
class ResidualStack(nn.Module):

  def __init__(self, channels, k=3):
    super().__init__()
    self.channels = channels

    self.res_1 = nn.Sequential(
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, k, padding=get_same_padding(3)),
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, k, padding=get_same_padding(3))
    )
    self.res_2 = nn.Sequential(
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, k, dilation=3, padding=get_same_padding(3, 3)),
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, k, padding=get_same_padding(3))
    )
    self.res_3 = nn.Sequential(
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, k, dilation=9, padding=get_same_padding(3, 9)),
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, k, padding=get_same_padding(3))
    )

    nn.utils.weight_norm(self.res_1[1])
    nn.utils.weight_norm(self.res_1[3])
    nn.utils.weight_norm(self.res_2[1])
    nn.utils.weight_norm(self.res_2[3])
    nn.utils.weight_norm(self.res_3[1])
    nn.utils.weight_norm(self.res_3[3])

  def forward(self, x):
    for l in [self.res_1, self.res_2, self.res_3]:
      r = l(x)
      x = x + r
    return x
  
  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.res_1[1])
    nn.utils.remove_weight_norm(self.res_1[3])
    nn.utils.remove_weight_norm(self.res_2[1])
    nn.utils.remove_weight_norm(self.res_2[3])
    nn.utils.remove_weight_norm(self.res_3[1])
    nn.utils.remove_weight_norm(self.res_3[3])

class ResidualBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride, resize='u'):
    super().__init__()

    self.in_channels  = in_channels
    self.out_channels = out_channels
    self.kernel_size  = kernel_size
    self.stride       = stride

    if resize == 'u':
        self.pre = weight_norm(ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=stride//2))
    else:
        self.pre = weight_norm(Conv1d         (in_channels, out_channels, kernel_size, stride, padding=stride//2))
    self.res_stack = ResidualStack(out_channels)

  def forward(self, x):
    x = F.leaky_relu(x, LRELU_SLOPE)
    #print('res_stack_x.shape:', x.shape)
    x = self.pre(x)
    #print('res_stack_pre.shape:', x.shape)
    x = self.res_stack(x)
    #print('res_stack_out.shape:', x.shape)
    return x

  def remove_weight_norm(self):
    remove_weight_norm(self.pre)
    self.res_stack.remove_weight_norm()

## All about HiFiGAN ##
class ResBlock(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        
        # 空洞卷积
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 卷积变换
            xt = c(xt)
            # 残差并入
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs: remove_weight_norm(l)

class ResBlock3(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 卷积变换
            xt = c(xt)
            # 残差并入
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs: remove_weight_norm(l)

class ResBlock_full(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)


'''Generator'''

## RetuneCNN
class Generator_RetuneCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_pre = weight_norm(Conv1d(1, 32, 15, padding=7))
        self.convs = nn.ModuleList([
            weight_norm(Conv1d( 32, 128, 41, padding=20)),
            weight_norm(Conv1d(128, 128, 41, padding=20)),
            weight_norm(Conv1d(128, 128, 41, padding=20)),
            weight_norm(Conv1d(128, 128, 41, padding=20)),
            weight_norm(Conv1d(128, 128, 41, padding=20)),
            weight_norm(Conv1d(128,  32, 41, padding=20)),
        ])
        self.conv_post = weight_norm(Conv1d(32, 1, 7, padding=3))

        self.conv_pre .apply(init_weights)
        self.convs    .apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, y):        # x(mel) is dummy
        y = self.conv_pre(y)
        for i in range(len(self.convs)):
            #y = F.leaky_relu(y, LRELU_SLOPE)
            y = torch.tanh(y)
            y = self.convs[i](y)
        #y = F.leaky_relu(y, LRELU_SLOPE)
        y = torch.tanh(y)
        y = self.conv_post(y)
        y = torch.tanh(y)
        
        return y

    def remove_weight_norm(self):
        for l in self.convs: remove_weight_norm(l)
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# MelGAN original
class Generator_MelGAN(nn.Module):

  def __init__(self, use_post=True):
    super().__init__()

    self.pre = weight_norm(Conv1d(hp.n_mel, 512, 7, 1, padding=get_same_padding(7)))
    self.res_blocks = nn.ModuleList([
        ResidualBlock(512, 256, 16, 8),
        ResidualBlock(256, 128, 16, 8),
        ResidualBlock(128,  64,  4, 2),
        ResidualBlock( 64,  32,  4, 2),
    ])
    self.post = weight_norm(Conv1d(32, 1, 7, 1, padding=get_same_padding(7))) if use_post else None

  def forward(self, x, y=None):
    x = self.pre(x)
    for l in self.res_blocks:
        x = l(x)
    if self.post:
        x = F.leaky_relu(x)
        x = self.post(x)
        x = torch.tanh(x)
    return x

  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.pre)
    nn.utils.remove_weight_norm(self.post)
    for l in self.res_blocks: l.remove_weight_norm()

# MelGAN fuse wav_ref at halfway
class Generator_MelGANRetune(nn.Module):

    def __init__(self):
        super().__init__()

        # for mel: let mel upsample 2 times (256 -> 128 -> 64)
        self.pre_x = weight_norm(Conv1d(hp.n_mel, 256, 7, 1, padding=get_same_padding(7)))
        self.ups_x = nn.ModuleList([
            ResidualBlock(256, 128, 16, 8),   # 256->128
            ResidualBlock(128,  64, 16, 8),   # 128->64
        ])

        # for wav_ref: let wav_ref downsample 2 times (1 -> 32 -> 64)
        self.pre_y = weight_norm(Conv1d(1, 16, 7, padding=get_same_padding(7)))
        self.downs_y = nn.ModuleList([
            ResidualBlock(16, 32, 4, 2, resize='d'),
            ResidualBlock(32, 64, 4, 2, resize='d'),
        ])

        # for merge
        self.alpha = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)

        # for decoder, upsample 2 times together
        self.ups_z = nn.ModuleList([
            ResidualBlock(128, 64, 4, 2),
            ResidualBlock( 64, 32, 4, 2),
        ])
        
        # for out
        self.post = weight_norm(Conv1d(32, 1, 7, padding=get_same_padding(7)))

        self.pre_x.apply(init_weights)
        self.pre_y.apply(init_weights)
        self.post .apply(init_weights)

    def forward(self, x, y_tmpl):
        DEBUG = False

        # decode mel
        x = self.pre_x(x)
        for l in self.ups_x:
            x = l(x)
            if DEBUG: print('x.shape:', x.shape)
        # NOTE: 64个点为一个周期

        # encode wav_ref
        y = self.pre_y(y_tmpl)
        #print('x.shape:', x.shape)
        for l in self.downs_y:
            y = l(y)
            if DEBUG: print('y.shape:', y.shape)

        # merge by concat
        #x, y = truncate_align(x, y)
        z = torch.cat([x, y * self.alpha], axis=1)    # concat in dim D
        # [16, 128, 2048]
        if DEBUG: print('z.shape:', z.shape)

        # decode z
        for l in self.ups_z:
            z = l(z)
            if DEBUG: print('z.shape:', z.shape)

        # got normalized wav!
        z = F.leaky_relu(z, LRELU_SLOPE)
        z = self.post(z)           # [B, 1, T]
        if DEBUG: print('z.shape:', z.shape)
        z = torch.tanh(z)

        return z

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups_x:   l.remove_weight_norm()
        for l in self.ups_z:   l.remove_weight_norm()
        for l in self.downs_y: l.remove_weight_norm()
        remove_weight_norm(self.pre_x)
        remove_weight_norm(self.pre_y)
        remove_weight_norm(self.post)

# MelGAN split c-v
class Generator_MelGANSplit(nn.Module):

    def __init__(self, ch=32):
        super().__init__()

        self.g_c = Generator_MelGAN(use_post=False)
        self.g_v = Generator_MelGAN(use_post=False)

        self.conv = weight_norm(Conv1d(ch, ch, 7, 1, padding=3))
        self.res_stack = ResidualStack(ch)
        self.post = weight_norm(Conv1d(ch, 1, 7, padding=get_same_padding(7)))

        self.conv.apply(init_weights)
        self.post.apply(init_weights)

    def forward(self, x_c, x_v, y_tmpl_c, y_tmpl_v, uv_ex):
        # gen code
        E_c = self.g_c(x_c)
        E_v = self.g_v(x_v)

        # mask & combine C-V
        uv_ex = uv_ex.unsqueeze(1)     # [B, T] -> [B, 1, T]
        E_c = E_c *      uv_ex
        E_v = E_v * (1 - uv_ex)
        z = E_c + E_v                   # [B, 32, T]

        # refine conjunction points
        z = self.conv(z)
        z = self.res_stack(z)

        # got normalized wav!
        z = F.leaky_relu(z, LRELU_SLOPE)
        z = self.post(z)           # [B, 1, T]
        z = torch.tanh(z)

        return z

    def remove_weight_norm(self):
        print('Removing weight norm...')
        self.g_c.remove_weight_norm()
        self.g_v.remove_weight_norm()
        self.res_stack.remove_weight_norm()
        remove_weight_norm(self.conv)
        remove_weight_norm(self.post)

# HiFiGAN original
class Generator_HiFiGAN(nn.Module):

    def __init__(self, use_post=True):
        super().__init__()

        self.num_kernels   = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(hp.n_mel, hp.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.upsample_rates, hp.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hp.upsample_initial_channel//(2**i), hp.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=k//2, output_padding=u-1)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(hp.resblock_kernel_sizes, hp.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3)) if use_post else None

        self.ups      .apply(init_weights)
        self.conv_pre .apply(init_weights)
        if self.conv_post: self.conv_post.apply(init_weights)

    def forward(self, x, y=None):       # y(wav) is dummy
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        if self.conv_post:
            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:         remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        if self.conv_post: remove_weight_norm(self.conv_post)

# HiFiGAN with single ResBlock_full
class Generator_HiFiGAN_mini(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_kernels   = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(hp.n_mel, hp.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.upsample_rates, hp.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hp.upsample_initial_channel//(2**i), hp.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=k//2, output_padding=u-1)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.upsample_initial_channel//(2**(i+1))
            self.resblocks.append(ResBlock_full(ch, 3, [1, 3, 9]))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.ups      .apply(init_weights)
        self.conv_pre .apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, y=None):       # y(wav) is dummy
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x = self.resblocks[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:         remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# HiFiGAN replacing ResBlock with single Conv
class Generator_HiFiGAN_micro(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_kernels   = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(hp.n_mel, hp.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.upsample_rates, hp.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hp.upsample_initial_channel//(2**i), hp.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=k//2, output_padding=u-1)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.upsample_initial_channel//(2**(i+1))
            self.resblocks.append(Conv1d(ch, ch, 41, 1, padding=20))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.ups      .apply(init_weights)
        self.conv_pre .apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, y=None):       # y(wav) is dummy
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x = self.resblocks[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:         remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# HiFiGAN no ResBlocks (only upsample)
class Generator_HiFiGAN_mu(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_kernels   = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(hp.n_mel, hp.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.upsample_rates, hp.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hp.upsample_initial_channel//(2**i), hp.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=k//2, output_padding=u-1)))
        self.conv_post = weight_norm(Conv1d(32, 1, 7, 1, padding=3))

        self.ups      .apply(init_weights)
        self.conv_pre .apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, y=None):       # y(wav) is dummy
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups: remove_weight_norm(l)
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# RefineGAN as in paper (7m)
class Generator_RefineGAN(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_kernels   = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.n_layer       = self.num_upsamples
        ch = 32

        # for wav_tmpl: T降采样256、D升采样256，基本不折损信息
        self.conv_pre_y = weight_norm(Conv1d(1, ch, 7, 1, padding=3))
        self.downs = nn.ModuleList([
            weight_norm(Conv1d(ch*2**i, ch*2**(i+1), k, u, padding=k//2))
            for i, (u, k) in enumerate(zip(hp.upsample_rates[::-1], hp.upsample_kernel_sizes[::-1]))
        ])
        # D = 64, 128, 256
        self.resblock = nn.ModuleList([     # 参考论文所述，encoder时只有一路
            ResBlock(ch * (2**(i+1)), 5, [1, 3])
            for i in range(len(self.downs))
        ])

        # for mel: [x:80->256]+[y:256] = [512,T]=>[1,256T]，冗余了两倍信息
        self.conv_pre = weight_norm(Conv1d(hp.n_mel, hp.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(ConvTranspose1d(hp.upsample_initial_channel//(2**i)*2, hp.upsample_initial_channel//(2**(i+1))*2, k, u, padding=k//2, output_padding=u-1))
            for i, (u, k) in enumerate(zip(hp.upsample_rates, hp.upsample_kernel_sizes))
        ])
        # D = 256, 128, 64
        self.resblocks = nn.ModuleList([
            ResBlock(hp.upsample_initial_channel//(2**i), k, d)
            for i in range(len(self.ups))
                for j, (k, d) in enumerate(zip(hp.resblock_kernel_sizes, hp.resblock_dilation_sizes))
        ])

        # for merge
        self.merge = nn.ModuleList([
            weight_norm(Conv1d(256+128, 256, 7, 1, padding=3)),
            weight_norm(Conv1d(128+ 64, 128, 7, 1, padding=3)),
            weight_norm(Conv1d( 64+ 32,  64, 7, 1, padding=3)),
        ])

        # for out
        self.conv_post = weight_norm(Conv1d(ch*2, 1, 7, 1, padding=3))

        self.noise = GaussianNoise().to(device)
        
    def forward(self, x, y):
        DEBUG = False
        def inspect(name, x):
            if DEBUG: print(f'{name}: {str(x.shape)}')

        o = [ ]

        ''' encoder '''
        y = self.conv_pre_y(y)
        inspect('prenet_y', y)
        for i, l in enumerate(self.downs):
            y = F.leaky_relu(y, LRELU_SLOPE)
            o.append(y)
            y = l(y)
            inspect('down_y', y)
            y = self.resblock[i](y)
            inspect('rb_y', y)

        ''' fuse '''
        x = self.conv_pre(x)
        inspect('prenet_x', x)
        #x, y = truncate_align(x, y)
        z = torch.cat([x, y], axis=1)      # concat by D
        inspect('fuse', z)

        ''' decoder '''
        for i in range(self.n_layer):
            z = F.leaky_relu(z, LRELU_SLOPE)
            z = self.ups[i](z)
            inspect('up_z', z)

            fm = o[self.n_layer-i-1]
            inspect('fm', fm)
            #z, fm = truncate_align(z, fm)          # in case of inference
            concat = torch.cat([z, fm], axis=1)    # concat by D
            inspect('concat_z', concat)
            z = self.merge[i](concat)
            inspect('merge_z', z)

            zs = 0
            z = self.noise(z)
            for j in range(self.num_kernels):
                zs += self.resblocks[i*self.num_kernels+j](z)
            z = zs / self.num_kernels
            z = self.noise(z)
            inspect('rb_z', z)

        # postnet: [ch=32, T] => [1, T], outval normalize
        z = F.leaky_relu(z, LRELU_SLOPE)
        z = self.conv_post(z)
        if DEBUG: inspect('posnet', z)
        z = torch.tanh(z)

        return z

    def remove_weight_norm(self):
        for l in self.downs:       remove_weight_norm(l)
        for l in self.ups:         remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)

# RefineGAN small size (2.1m)
class Generator_RefineGAN_small(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_kernels   = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.n_layer       = self.num_upsamples
        ch = 32

        # for wav_tmpl: T降采样256倍、D升采样128倍，总体上折损一半的信息
        # D = 16
        self.conv_pre = weight_norm(Conv1d(1, ch//2, 7, 1, padding=3))
        # D = 32, 64, 128
        self.downs = nn.ModuleList([
            weight_norm(Conv1d(ch*2**i//2, ch*2**(i+1)//2, k, u, padding=k//2))
                for i, (u, k) in enumerate(zip(hp.upsample_rates[::-1], hp.upsample_kernel_sizes[::-1]))
        ])
        # D = 32, 64, 128
        self.resblock = nn.ModuleList([     # 参考论文所述，encoder时只有一路
            ResidualStack(ch * 2**i)
                for i in range(len(self.downs))
        ])

        # for mel: 从[80,T]=>[1,256T], 总体需要脑补1.23倍的信息，全靠NN自动插值（注意：原始MelGAN从[80,T]=>[1,256T], 需要脑补3.2倍
        # D = 80 + 128 -> 256
        self.conv_fuse = weight_norm(Conv1d(hp.n_mel + hp.upsample_initial_channel//2, hp.upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(ConvTranspose1d(hp.upsample_initial_channel//(2**i), hp.upsample_initial_channel//(2**(i+1)), k, u, padding=k//2, output_padding=u-1))
                for i, (u, k) in enumerate(zip(hp.upsample_rates, hp.upsample_kernel_sizes))
        ])
        # D = 128, 64, 32
        #self.resblocks = nn.ModuleList([
        #    ResBlock(hp.upsample_initial_channel//(2**(i+1)), k, d)
        #        for i in range(len(self.ups))
        #            for j, (k, d) in enumerate(zip(hp.resblock_kernel_sizes, hp.resblock_dilation_sizes))
        #])
        # NOTE: 9-3-1 比 1-3-9 的stft_loss下降快
        self.resblocks = nn.ModuleList([
            ResBlock3(128, 3, [9, 3, 1]), ResBlock3(128, 5, [9, 3, 1]), ResBlock3(128, 7, [9, 3, 1]),
            ResBlock3( 64, 3, [9, 3, 1]), ResBlock3( 64, 5, [9, 3, 1]), ResBlock3( 64, 7, [9, 3, 1]),
            ResBlock3( 32, 3, [9, 3, 1]), ResBlock3( 32, 5, [9, 3, 1]), ResBlock3( 32, 7, [9, 3, 1]),
        ])

        # for merge hidden from encoder
        self.merge = nn.ModuleList([
            weight_norm(Conv1d(128+64, 128, 7, 1, padding=3)),
            weight_norm(Conv1d( 64+32,  64, 7, 1, padding=3)),
            weight_norm(Conv1d( 32+16,  32, 7, 1, padding=3)),
        ])

        # for out
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # for stability
        self.noise = GaussianNoise().to(device)

        self.conv_pre .apply(init_weights)
        self.conv_fuse.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.downs    .apply(init_weights)
        self.merge    .apply(init_weights)
        self.ups      .apply(init_weights)
        
    def forward(self, x, y):
        DEBUG = False
        def inspect(name, x):
            if DEBUG: print(f'{name}: {str(x.shape)}')

        o = [ ]     # 存放参考波形的一些降采样版本(encoder hidden)

        ''' encoder '''
        y = self.conv_pre(y)
        inspect('prenet_y', y)
        for i, l in enumerate(self.downs):
            y = F.leaky_relu(y, LRELU_SLOPE)
            o.append(y)
            y = l(y)
            inspect('down_y', y)
            y = self.resblock[i](y)
            inspect('rb_y', y)

        ''' fuse '''
        y = F.leaky_relu(y, LRELU_SLOPE)
        #x, y = truncate_align(x, y)
        z = torch.cat([x, y], axis=1)      # concat by D
        inspect('concat', z)
        z = self.conv_fuse(z)
        inspect('fuse', z)

        ''' decoder '''
        for i in range(self.n_layer):
            z = F.leaky_relu(z, LRELU_SLOPE)
            z = self.ups[i](z)
            inspect('up_z', z)

            fm = o[self.n_layer-i-1]
            inspect('fm', fm)
            #z, fm = truncate_align(z, fm)          # in case of inference
            concat = torch.cat([z, fm], axis=1)    # concat by D
            inspect('concat_z', concat)
            z = self.merge[i](concat)
            inspect('merge_z', z)

            zs = 0
            z = self.noise(z)
            for j in range(self.num_kernels):
                zs += self.resblocks[i*self.num_kernels+j](z)
            z = zs / self.num_kernels
            z = self.noise(z)
            inspect('rb_z', z)

        # postnet: [ch=32, T] => [1, T], outval normalize
        z = F.leaky_relu(z, LRELU_SLOPE)
        z = self.conv_post(z)
        if DEBUG: inspect('posnet', z)
        z = torch.tanh(z)

        return z

    def remove_weight_norm(self):
        for l in self.downs:       remove_weight_norm(l)
        for l in self.ups:         remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_fuse)
        nn.utils.remove_weight_norm(self.conv_post)

# HiFiGAN fuse wav_ref at halfway
class Generator_HiFiGANRetune(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_kernels   = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        
        # for mel
        self.pre_x = weight_norm(Conv1d(hp.n_mel, 256, 7, 1, padding=3))
        self.ups_x = nn.ModuleList([
            weight_norm(ConvTranspose1d(256, 128, 16, 8, padding=(16-8)//2)),
            weight_norm(ConvTranspose1d(128,  64, 16, 8, padding=(16-8)//2)),
        ])
        self.resblocks_x = nn.ModuleList([
            ResBlock(128, 3, [1, 2]), ResBlock(128, 5, [2, 6]), ResBlock(128, 7, [3, 12]), 
            ResBlock( 64, 3, [1, 2]), ResBlock( 64, 5, [2, 6]), ResBlock( 64, 7, [3, 12]), 
        ])

        # for wav_ref
        self.pre_y = weight_norm(Conv1d(1, 16, 7, 1, padding=3))
        self.downs_y = nn.ModuleList([
            weight_norm(Conv1d(16, 32, 4, 2, padding=(4-2)//2)),
            weight_norm(Conv1d(32, 64, 4, 2, padding=(4-2)//2)),
        ])
        self.resblock_y = nn.ModuleList([
            ResBlock_full(32, 3, [1, 3, 9]),
            ResBlock_full(64, 3, [1, 3, 9]),
        ])

        # for merge
        self.alpha = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)

        self.ups_z = nn.ModuleList([
            weight_norm(ConvTranspose1d(128, 64, 4, 2, padding=(4-2)//2)),
            weight_norm(ConvTranspose1d( 64, 32, 4, 2, padding=(4-2)//2)),
        ])
        self.resblocks_z = nn.ModuleList([
            ResBlock(64, 3, [1, 2]), ResBlock(64, 5, [2, 6]), ResBlock(64, 7, [3, 12]), 
            ResBlock(32, 3, [1, 2]), ResBlock(32, 5, [2, 6]), ResBlock(32, 7, [3, 12]), 
        ])

        # for out
        self.post = weight_norm(Conv1d(32, 1, 7, 1, padding=3))

        self.pre_x  .apply(init_weights)
        self.ups_x  .apply(init_weights)
        self.pre_y  .apply(init_weights)
        self.downs_y.apply(init_weights)
        self.ups_z  .apply(init_weights)
        self.post   .apply(init_weights)

    def forward(self, x, y_tmpl):       # y(wav) is dummy
        DEBUG = False
        
        x = self.pre_x(x)
        for i in range(len(self.ups_x)):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups_x[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks_x[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        if DEBUG: print('x.shape:', x.shape)

        y = self.pre_y(y_tmpl)
        for i in range(len(self.downs_y)):
            y = F.leaky_relu(y, LRELU_SLOPE)
            y = self.downs_y[i](y)
            y = self.resblock_y[i](y)
        if DEBUG: print('y.shape:', y.shape)

        z = torch.cat([x, y * self.alpha], axis=1)
        if DEBUG: print('z.shape:', z.shape)

        for i in range(len(self.ups_z)):
            z = F.leaky_relu(z, LRELU_SLOPE)
            z = self.ups_z[i](z)
            zs = 0
            for j in range(self.num_kernels):
                zs += self.resblocks_z[i*self.num_kernels+j](z)
            z = zs / self.num_kernels
        if DEBUG: print('z.shape:', z.shape)

        z = F.leaky_relu(z)
        z = self.post(z)
        z = torch.tanh(z)

        return z

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups_x:   remove_weight_norm(l)
        for l in self.downs_y: remove_weight_norm(l)
        for l in self.ups_z:   remove_weight_norm(l)
        for l in self.resblocks_x:   l.remove_weight_norm()
        for l in self.resblocks_z:   l.remove_weight_norm()
        remove_weight_norm(self.pre_x)
        remove_weight_norm(self.pre_y)
        remove_weight_norm(self.post)

# HiFiGAN split c-v
class Generator_HiFiGANSplit(nn.Module):

    def __init__(self, ch=32):
        super().__init__()

        self.g_c = Generator_HiFiGAN(use_post=False)
        self.g_v = Generator_HiFiGAN(use_post=False)

        self.num_kernels = len(hp.resblock_kernel_sizes)
        self.conv = weight_norm(Conv1d(ch, ch, 7, 1, padding=3))
        self.post = weight_norm(Conv1d(ch, 1, 15, padding=7))

        self.conv.apply(init_weights)
        self.post.apply(init_weights)
        
    def forward(self, x_c, x_v, y_tmpl_c, y_tmpl_v, uv_ex):
        # generate C-V seperately
        E_c = self.g_c(x_c)   # [B, 32, T]
        E_v = self.g_v(x_v)   # [B, 32, T]

        # mask & combine C-V
        uv_ex = uv_ex.unsqueeze(1)     # [B, T] -> [B, 1, T]
        E_c = E_c *      uv_ex
        E_v = E_v * (1 - uv_ex)
        z = E_c + E_v                   # [B, 32, T]

        # refine conjunction points
        z = self.conv(z)                # [B, 32, T]

        # got normalized wav!
        z = F.leaky_relu(z, LRELU_SLOPE)
        z = self.post(z)           # [B, 1, T]
        z = torch.tanh(z)

        return z

    def remove_weight_norm(self):
        print('Removing weight norm...')
        self.g_c.remove_weight_norm()
        self.g_v.remove_weight_norm()
        remove_weight_norm(self.conv)
        remove_weight_norm(self.post)
