# TransTacoS-RetuneGAN

    A lighter-weight (perhaps!) Text-to-Speech for Chinese/Mandarin synthesize, inspired by Tacotron & FastSpeech2 & RefineGAN.
    It is also my shitty graduation design project, just a toy, so lower your expectations :)

----

## Quick Start

### setup

Since `TransTacoS` is implemented in tensorflow while `RefineGAN` in torch respectively, you could separate them by creating virtual envs, but they are likely not to conflict, thus you could try to put all these together:

  - install `tensorflow-gpu==1.14.0 tensorboard==1.14.0` following `https://tensorflow.google.cn/install/pip`
  - install `torch==1.8.0+cu1xx torchaudio==0.8.0` following `https://pytorch.org/`, where `cu1xx` is your cuda version
  - run `pip install -r requirements.txt` for the rest dependencies

### dataset

  - download and unzip the open source dataset [DataBaker](https://www.data-baker.com/data/index/TNtts)
  - other dataset requires user-defined preprocessor, please refer to `transtacos/dataset/__skel__.py`

### train

  - check path configs in all `Makefile`
  - `cd transtacos & make preprocess` to prepare acoustic features (linear/mel/f0/c0/zcr)
  - `cd transtacos & make train` to train TransTacoS
  - `cd retunegan & make finetune` to train RetuneGAN using preprocessed linear spectrograms  (rather than from raw wave)

### deploy

  - check port configs in all `Makefile`
  - `cd transtacos & make server` to start TransTacoS headless HTTP server (default at port 5105)
  - `cd retunegan & make server` to start RetuneGAN headless HTTP server (default at port 5104)
  - `python app.py` to start the WebUI app (default at port 5103)
  - point your browser to `http://localhost:5103`, now have a try!

## Model Architecture

### TransTacoS

![align](/img/tts_out_align.png)

![spec](/img/tts_out_spec.png)

#### What I actually did:

- TransTacoS := 
  - embed: self-designed G2P solution (syl4), employ prodosy marks as linguistic feature
    - for G2P solution, I tried char(seq), phoneme, pinyin (CV), CVC, VCV, CVVC ...
      - de facto, they merely influence mel_loss, but effect that how controllable the pronounciation is
      - so I design **syl4** to split phoneme against tone (just small improvement)
    - I predict prodosy marks by simple CNN rather than RNN, it's not enough reasonable though..
  - encoder: modified from FastSpeech2
    - I use multi-head self-DotAttn with GFFW (gated-FFW) for backbone transform
    - f0/c0 feature is quantilized to embed, then fused in with cross-DotAttn
  - decoder: inherited from Tacotron
    - this RNN+LSA decoder is really complicated, I dare not to touch :(
  - posnet: self-designed simple MLP and grouped-Linear
    - I found simple Linear layer with growing depth leads to lower mel_loss than Conv1d
    - thus realy DO NOT understand why Tacotron2 even FastSpeech2 still use a postnet directly strech n_channels from 80 (n_mel) to 1025 (n_freq), this is surely tooooo hard to compensate information loss

Frankly speaking, TransTacoS didn't improve any thing profoundly from Tacotron, but I just found that shallower network leads to lower mel_loss, so maybe simple embed+decoder is already enough :(

#### Tips of ideas to try or failed: 

- To predict `f0/sp/ap` features so that we can use WORLD vocoder
  - It seems `sp` is OK, because it resembles mel very much
  - but `ap` requires to be carefully normalized, and accurate `f0` is even harder to predict
  - audio quality of WORLD sounds worse than Griffin-Lim at times :(
- To predict spectrograms' magnitude part together with phase part, so that we can directly vocode using pure `istft`
  - It should be hard to optimize losses on phase, especially on conjunction points between mel frames
  - but these guys claim that they did it: [iSTFTNet](https://arxiv.org/abs/2203.02395)
- To predict `f0` and `dyn` so that vocoder might benefits
  - I don't know how to separate them from mel, because so far I must regularize decoder's output to be mel (for the sake of teacher force)
  - When I remove the mel loss, I found that the RNN decoder became lazy to learn, and the align model also not work
  - mel is even more quantized that linear, to abstract `f0` and `dyn` from only mel seems not the reasonable
- To extract duration info from the soft-aligned alignment map, so that we can further train a non-autogressive FastSpeech2 to speed up inference
  - just like [FastSpeech](https://arxiv.org/abs/1905.09263) and [DeepSinger](https://arxiv.org/abs/2007.04590) did
  - but FastSpeech reported that extracted duration is not enough accurate, yet I could not fully understand and reproduce DeepSinger's DP algorithm for duration extraction


### RetuneGAN

![gen_wav_cmp](/img/rtg_cmp.png)


#### What I actually did:

- RetuneGAN :=
  - preprocess
    - extract `reference wav` using Griffin-Lim
    - extract `u/v mask` by hand-tuned zrc/c0 threshold (for Split-G only)
  - generators
    - `UNet-G` (encoder-decoder generator): modified from RefineGAN, we use the output of Griffin-Lim as reference wav, rather than an F0/C0-guided hand-crafted *speech template*
    - `Split-G` (split u/v generator): self-designed, inspired by Multi-Band MelGAN, but I found the generated quality is holy shit :(
      - `ResStack` borrowed from MelGAN
      - `ResBlock` modified from HiFiGAN
  - discriminators
    - `MSD` (multi scale discriminator): borrowed from MelGAN, I think it's good for plosive consonants
    - `MPD` (multi period discriminator): borrowed from HiFiGAN, I take it as a multiple MSDs' stack-up
    - `MTD` (multi stft discriminator): modified from UnivNet, it has two work modes depending on its input (MPSD seems better indeed ...)
      - `MPSD` (multi parameter spectrogram discriminator): like in UnivNet, but we let it judge both phase part and magnitude part
      - `PHD` (phase discriminator): self-designed, care more about phase, since `l_mstft` has already regulated magnitude
        - input of MPSD is `[(mag_real, phase_real), (mag_fake, phase_fake)]`, thus distinguishes real/fake stft data
        - input of PHD  is `[(mag_real, phase_real), (mag_real, phase_fake)]`, thus ONLY distinguishes real/fake phase
  - losses
    - `l_adv` (adversarial loss): modified from HiFiGAN, but relativized
    - `l_fm` (feature map loss): borrowed from MelGAN
    - `l_mstft` (multi stft loss): modified from Parallel WaveGAN, but we calculate mel_loss rather than linear_loss
    - `l_env` (envlope loss): borrowed from RefineGAN
    - `l_dyn` (dynamic loss): self-designed, inspired by `l_env` 
    - `l_sm` (strip mirror loss): self-designed, but might hurts audio quality :(

Oh my dude, it's really a biggy feng-he monster :( 

#### Tips of ideas to try or failed: 

- To divide *consonant* part against *vowel* part in time domain, then use two generator to generate them separately
  - I found this will bring breakups in conjunction points, thus audio sounds noisy, yet mstft loss will be more unstable
- To shallow fuse the reference wav with half-decoded mel, rather than the `encode-merge-decode` UNet architecture
  - I found this will make the generator lazy to learn, even overfit to train set
  - might because waveform is far from audio semantics but near representation, so an encoder is necessary to extract semantical info
- NOTE: the weight of mstft loss should NOT be too overwhelming in front of adversarial loss (like in HiFiGAN)
  - adversarial loss leads to more clear plosives (`b/p/g/k/d/t`), while `mstft loss` contributes little to consonants
  - `hop_length` in stft is much larger that `stride` in discriminators, thus mstft loss is usually more coarse than adversarial loss in time domain 

## Acknowledgements

Codes referred to:

- [keithito's Tacotron](https://github.com/keithito/tacotron)
- [jaywalnut310's MelGAN](https://github.com/jaywalnut310/MelGAN-Pytorch)
- [jik876's official HiFiGAN](https://github.com/jik876/hifi-gan)
- [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2/)

Ideas plagiarized from:

  - [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
  - [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
  - [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
  - [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
  - [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106)
  - [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
  - [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889)
  - [RefineGAN: Universally Generating Waveform Better than Ground Truth with Highly Accurate Pitch and Intensity Responses](https://arxiv.org/abs/2111.00962)

code release kept under the MIT license, greatest thanks all the authors!! :)

----

by Armit
2022/02/15
2022/05/25 