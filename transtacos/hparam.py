# Text
g2p                = 'syl4'   # ['seq', 'syl4']

# Audio
sample_rate        = 22050    # sample rate (Hz) of wav file
n_fft              = 2048
win_length         = 1024     # :=n_fft//2
hop_length         = 256      # :=win_length//4, 11.6ms, 平均1个拼音音素对应9帧(min2~max20)
n_mel              = 80       # MEL谱段数 (default: 160), 120 should be more reasonable
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

# Model
outputs_per_step    = 5       # default: 5 (aka. reduction factor r), 某种意义上的韵律量化，r越小韵律越精细、但rnn可能记不住长序列
                              # 一般来说r取半音素的平均长度，这对元音连接类似于VCV
hidden_gauss_std    = 1e-5

embed_depth         = 256     # text/prds embed depth
var_embed_depth     = 64      # f0/c0 embed depth
posenc_depth        = 32      # pos_enc embed depth
txt_use_posenc      = True    # 不需要PE好像才能学出sa的对角线(?)
var_use_posenc      = True    # 需要PE才能学出ca的对角线
prdsnet_depth       = 64
prdsnet_conv_k      = 9
embed_dropout       = False

encoder_depth       = 256     # aka. inner_repr_depth
encoder_type        = 'sa'    # ['sa', 'cb']
if encoder_type == 'sa':      # like FastSpeech2
  encoder_attn_layers = 2     # NOTE: set 4 will lead to nan in loss (grad vanish?)
  encoder_attn_nhead  = 2
  encoder_dropout     = False
  encoder_fusenet     = True
  gffw_conv_k         = 9
  var_prednet_depth   = 64
  var_prednet_conv_k  = 13
if encoder_type == 'cb':      # like Tacotron
  encoder_conv_K      = 16
  highway_layers      = 4

decoder_layers      = 2       # single layer is not enough
decoder_depth       = 512     # default: 1024
attention_depth     = 128     # single LSA
prenet_depths       = [256]   # prenet for decoder RNN, single layer seems enough
decoder_sew_layer   = False

n_mel_low           = 42
posnet_depth        = 512
posnet_ngroup       = 8

# Training
max_steps             = 320000   # force stop train
max_ckpt              = 1
batch_size            = 16
adam_beta1            = 0.9
adam_beta2            = 0.999    # 0.98
adam_eps              = 1e-7
reg_weight            = 1e-6     # 1e-8
sim_weight            = 1e-5
initial_learning_rate = 0.001
decay_learning_rate   = True     # decrease learning rate by step, see `models.tacotron._learning_rate_decay()`
tf_method             = 'mix'    # ['random', 'mix', 'force']
tf_init               = 1.0
tf_start_decay        = 20000    # default: 20000
tf_decay              = 200000   # default: 200000

# Eval
max_iters           = 300      # max iter of RNN, 最多产生max_iters*r帧、防止无限生成
gl_iters            = 30       # griffin_lim algorithm iters (default: 60)
gl_power            = 1.2      # Power to raise magnitudes to prior to Griffin-Lim
postprocess         = False    # see `audio.save_wav()`

# MISC
randseed            = 114514
debug               = False
