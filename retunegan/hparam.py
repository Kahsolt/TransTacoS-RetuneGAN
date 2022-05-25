'''Audio: proxy by trastacos, plz keep sync'''
# Audio
sample_rate        = 22050    # sample rate (Hz) of wav file
n_fft              = 2048
win_length         = 1024     # :=n_fft//2
hop_length         = 256      # :=win_length//4, 11.6ms, 平均1个拼音音素对应9帧(min2~max20)
n_mel              = 80       # MEL谱段数 (default: 160), 120 should be more reasonable though
n_freq             = 1025     # 线性谱段数 :=n_fft//2+1
preemphasis        = 0.97     # 增强高频，使EQ均衡
ref_level_db       = 20       # 最高考虑的谱幅值(虚拟0dB)，理论上安静环境下取94，但实际上录音越嘈杂该值应越小 (default: 20)
min_level_db       = -100     # 最低考虑的谱幅值，用于动态范围截断压缩 (default: -100)
max_abs_value      = 4        # 将谱幅值正则化到 [-max_abs_value, max_abs_value]
trim_below_peak_db = 35       # trim beginning/ending silence parts (default：60)
fmin               = 125      # MEL滤波器组频率上下限 (set 55/3600 for male)
fmax               = 7600
rf0min             = 'D2'     # 基频检测上下限
rf0max             = 'D5'

## see `Databaker stats` or `stats.txt` in preprocessed folder
c0min              = 4.6309418394230306e-05
c0max              = 0.3751049339771271
f0min              = 73.25581359863281
f0max              = 595.9459228515625
n_tone             = 5+1
n_prds             = 5+1
n_c0_bins          = 32
n_f0_bins          = None     # keep None for auto detect using f0min & f0max
n_f0_min           = None     # as offset
maxlen_text        = 128      # for pos_enc, 27 in train set
maxlen_spec        = 1024     # for pos_enc, 524 in train set

####################################################################################################################

'''Audio'''
segment_size          = 8192
window_fn             = 'hann'        # ['bartlett', 'blackman', 'hamming', 'hann', 'kaiser']
mel_scale             = 'slaney'      # 'htk' is smooth, 'slaney' breaks at f=1000; see `https://blog.csdn.net/qq_39995815/article/details/116269040`
gl_iters              = 4
gl_momentum           = 0.7           # 0.99 may deadloop
gl_power              = 1.2           # power magnitudes before Griffin-Lim
ref_wav               = 'y'           # ['y', 'dy']


'''Model'''
# RetuneCNN       (        |          | mstft=       at 30 epoch)
# HiFiGAN_mini    (        |          | mstft=       at 30 epoch)
# HiFiGAN_micro   (        |          | mstft=       at 30 epoch)
# HiFiGAN_mu      (        |          | mstft=       at 30 epoch)
# RefineGAN       (        |          | mstft=       at 30 epoch)
# RefineGAN_small (2748371 |          | mstft=       at 30 epoch)
# MelGAN          (4524290 | 2.36 s/b | mstft=10.084 at 30 epoch)
# MelGANRetune    (1409427 | 2.42 s/b | mstft= 7.000 at 30 epoch)
# MelGANSplit
# HiFiGAN         (1421314 | 2.30 s/b | mstft=10.346 at 30 epoch)
# HiFiGANRetune   (1716627 | 2.45 s/b | mstft= 7.041 at 30 epoch)   # 高频撕裂
# HiFiGANSplit    (2849890 | 2.49 s/b | mstft=11.320 at 30 epoch)

# generator
generator_ver             = 'RefineGAN_small'
split_cv                  = generator_ver.endswith('Split')
upsample_rates            = [8, 8, 4]
upsample_kernel_sizes     = [15, 15, 7]
upsample_initial_channel  = 256
resblock_kernel_sizes     = [3, 5, 7]
resblock_dilation_sizes   = [[1, 2], [2, 6], [3, 12]]
#resblock_kernel_sizes     = [3, 7, 11]
#resblock_dilation_sizes   = [[2, 3], [3, 5], [5, 11]]

# discriminator
msd_layers                = 3
mpd_periods               = [3, 5, 7, 11]
multi_stft_params         = [
  # (n_fft, win_length, hop_length)
  # (2048, 1024, 256),
  # (1024,  512, 128),
  # ( 512,  256,  64),
  # by UnivNet
  (2048, 1024, 240),
  (1024,  512, 120),
  ( 512,  256,  60),
]
phd_layers          = len(multi_stft_params)
phd_input           = 'stft'       # ['phase', 'stft'], phase似乎不太行

# loss
relative_gan_loss   = False
strip_mirror_loss   = False
dynamic_loss        = True
envelope_loss       = False
envelope_pool_k     = 160       # use `tools.test_envolope.py` to pickle proper value
downsample_pool_k   = 4


'''Misc'''
from sys import platform
debug               = platform == 'win32'     # debug locally on Windows, actually train on Linux
randseed            = 114514


'''Training'''
num_workers         = 1 if debug else 4
batch_size          = 4 if debug else 16      # 16 = 567.6 steps per epoch
learning_rate_d     = 2e-4
learning_rate_g     = 1.8e-4
d_train_times       = 2         # 更新一次G的同时更新多少次D
adam_b1             = 0.8
adam_b2             = 0.99
lr_decay            = 0.999

w_loss_fm           = 2
w_loss_mstft        = 8
w_loss_env          = 4
w_loss_dyn          = 4
w_loss_sm           = 0.01


'''Eval'''
valid_limit         = batch_size * 4
