import os
from random import randint

import numpy as np
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt

import hparam as hp
import audio as A

# for fintune tune with TranTacoS
from audio_proxy import AP


assert hp.segment_size % hp.hop_length == 0
frames_per_seg = hp.segment_size // hp.hop_length


class Dataset(Dataset):
    
    def __init__(self, name, data_dp, finetune=False, limit=None):
        self.is_train = name == 'train'
        self.data_dp = data_dp        # the preprocessed folder containing index files and mel/mag features (if finetune)
        self.finetune = finetune

        with open(os.path.join(data_dp, 'wav_path.txt')) as fh:
            wav_path = fh.read().strip()
        with open(os.path.join(data_dp, f'{name}.txt'), encoding='utf-8') as fh:
            self.wav_fps = [os.path.join(wav_path, line.split('|')[0] + '.wav') for line in fh.readlines() if line]
            if limit: self.wav_fps = self.wav_fps[:limit]

        self.data = [None] * len(self.wav_fps)

    def __len__(self):
        return len(self.wav_fps)
    
    def __getitem__(self, index):
        # repreprocess & cache
        if self.data[index] is None:
            ''' prepare GT wav '''
            wav_fp = self.wav_fps[index]
            if not self.finetune:
                wav = A.load_wav(wav_fp)
                if self.is_train:
                    # aug data once and freeze
                    wav = A.augment_wav(wav)
                wav = A.align_wav(wav)
            else:
                # keep identical to `preprocessor.make_metadata()` of TransTacoS
                wav = AP.load_wav(wav_fp)
                wav = AP.trim_silence(wav)
                wav = AP.align_wav(wav)
            
            wavlen = len(wav)
            
            ''' prepare GT mel '''
            if not self.finetune:
                # `[:-1]` to avoid extra tailing frame
                mag = A.get_mag(wav[:-1])   # [M, T]
            else:
                # keep identical to preprocessors of TransTacoS
                name = os.path.splitext(os.path.basename(wav_fp))[0]
                mag = np.load(os.path.join(self.data_dp, f'mag-{name}.npy'))   # [M, T]
                mag = AP.spec_to_natural_scale(mag)

            mel = A.mag_to_mel(mag)

            if self.is_train:
                # aug data once and freeze
                mel_aug = A.augment_spec(mel, rounds=5)
                mel = mel / 2 + mel_aug / 2

            ''' prepare ref wav '''
            try:
                wav_tmpl = A.inv_mag(mag, wavlen=wavlen-1)   # `wavlen-1` to avoid extra tailing frame
                wav_tmpl = np.pad(wav_tmpl, (0, 1))          # pad to align
            except:
                breakpoint()
            
            # dy: 按理说一阶差分可以大致显示脉冲位置，但是wav_tmpl的相位可能有点差
            if hp.ref_wav == 'dy':
                wav_tmpl = np.pad(wav_tmpl, (0, 1))
                wav_tmpl = np.asarray([b-a for a, b in zip(wav_tmpl[:-1], wav_tmpl[1:])])

            ''' prepare u/v mask '''
            if hp.split_cv:                             # 时域法误差有点大
                zcr = A.get_zcr(wav_tmpl[:-1])
                dyn = A.get_c0 (wav_tmpl[:-1])
                uv  = A.get_uv (zcr, dyn)
        
            ''' prepare u/v-splitted mel & ref wav '''
            if hp.split_cv:
                uv_ex = np.repeat(uv, hp.hop_length)
                wav_tmpl_c = wav_tmpl *      uv_ex
                wav_tmpl_v = wav_tmpl * (1 - uv_ex)
                mel_min   = mel.min()
                mel_shift = mel - mel_min       # assure > 0 for mask product
                mel_c = mel_shift *    uv  + mel_min
                mel_v = mel_shift * (1-uv) + mel_min
            
            if not 'check':
                if hp.split_cv:
                    wav_c = wav *      uv_ex
                    wav_v = wav * (1 - uv_ex)
                    plt.subplot(411); plt.plot(wav_c,      'r') ; plt.plot(wav_v,      'b')
                    plt.subplot(412); plt.plot(wav_tmpl_c, 'r') ; plt.plot(wav_tmpl_v, 'b')
                    plt.subplot(413); sns.heatmap(mel_c, cbar=False) ; plt.gca().invert_yaxis()
                    plt.subplot(414); sns.heatmap(mel_v, cbar=False) ; plt.gca().invert_yaxis()
                    plt.show()
                else:
                    plt.subplot(411); plt.plot(wav, 'b')
                    plt.subplot(412); sns.heatmap(mag, cbar=False) ; plt.gca().invert_yaxis()
                    plt.subplot(413); plt.plot(wav_tmpl, 'r')
                    plt.subplot(414); sns.heatmap(mel, cbar=False) ; plt.gca().invert_yaxis()
                    plt.show()
            
            ''' check shape aligns '''
            if hp.split_cv: assert len(dyn) == len(zcr) == mel.shape[1]
            assert len(wav) == len(wav_tmpl) == mel.shape[1] * hp.hop_length

            ''' done '''
            if hp.split_cv: 
                self.data[index] = (mel, wav, mel_c, mel_v, wav_tmpl_c, wav_tmpl_v, uv_ex)
            else:
                self.data[index] = (mel, wav, wav_tmpl)
        
        # get from cache (full length data)
        if hp.split_cv: 
            mel, wav, mel_c, mel_v, wav_tmpl_c, wav_tmpl_v, uv_ex = self.data[index]
        else:
            mel, wav, wav_tmpl = self.data[index]

        # make slices during training: wav[S=8192] <=> mel[T=32]
        if self.is_train:
            wavlen, mellen = len(wav), mel.shape[1]
            if wavlen > hp.segment_size:
                cp = randint(0, mellen - frames_per_seg - 1)
                if hp.split_cv:
                    mel_c      = mel_c     [:, cp : cp + frames_per_seg]
                    mel_v      = mel_v     [:, cp : cp + frames_per_seg]
                    wav_tmpl_c = wav_tmpl_c[cp * hp.hop_length : (cp + frames_per_seg) * hp.hop_length]
                    wav_tmpl_v = wav_tmpl_v[cp * hp.hop_length : (cp + frames_per_seg) * hp.hop_length]
                    wav        = wav       [cp * hp.hop_length : (cp + frames_per_seg) * hp.hop_length]
                    uv_ex      = uv_ex     [cp * hp.hop_length : (cp + frames_per_seg) * hp.hop_length]
                else:
                    mel        = mel       [:, cp : cp + frames_per_seg]
                    wav        = wav       [cp * hp.hop_length : (cp + frames_per_seg) * hp.hop_length]
                    wav_tmpl   = wav_tmpl  [cp * hp.hop_length : (cp + frames_per_seg) * hp.hop_length]
            else:
                if hp.split_cv:
                    mel_c      = np.pad(mel_c, (0, 0, 0, frames_per_seg - mellen), mel.min())
                    mel_v      = np.pad(mel_v, (0, 0, 0, frames_per_seg - mellen), mel.min())
                    wav_tmpl_c = np.pad(wav_tmpl_c, (0, hp.segment_size - wavlen))
                    wav_tmpl_v = np.pad(wav_tmpl_v, (0, hp.segment_size - wavlen))
                    wav        = np.pad(wav,        (0, hp.segment_size - wavlen))
                    uv_ex      = np.pad(uv_ex,      (0, hp.segment_size - wavlen))
                else:
                    mel        = np.pad(mel,   (0, 0, 0, frames_per_seg - mellen), mel.min())
                    wav        = np.pad(wav,        (0, hp.segment_size - wavlen))
                    wav_tmpl   = np.pad(wav_tmpl,   (0, hp.segment_size - wavlen))

        # mel:      由外源mag滤波而来，作为推断时的主要输入
        # wav_tmpl: 基于外源mag用传统算法得到的粗糙波形，作为推断时的参考输入
        # wav:      真实的目标录音波形，作为训练时的目标输出
        # uv_ex:    清音掩码，可作为推断时的参考输入
        if hp.split_cv:
            ret = mel_c, mel_v, wav_tmpl_c, wav_tmpl_v, wav, uv_ex
        else:
            ret = mel, wav_tmpl, wav

        return [x.astype(np.float32) for x in ret]
