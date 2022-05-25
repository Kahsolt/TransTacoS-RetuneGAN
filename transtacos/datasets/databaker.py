#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/01/10 

import os
from re import compile as Regex
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import hparam as hp
import audio as A

DROPOUT_2SIGMA = True


# identically collected from `DataBaker`
PUNCT_KANJI_REGEX = Regex(r'，|。|、|：|；|？|！|（|）|“|”|…|—')


def preprocess(args) -> Tuple[List[Tuple], dict]:
  wav_dp = os.path.join(args.base_dir, 'DataBaker', 'Wave')
  out_dp = os.path.join(args.base_dir, args.out_dir)
  os.makedirs(out_dp, exist_ok=True)
  label_dict = parse_label_file(os.path.join(args.base_dir, 'DataBaker', 'ProsodyLabeling', '000001-010000.txt'))

  executor = ProcessPoolExecutor(max_workers=args.num_workers)
  futures = []
  for name, feats in label_dict.items():
    wav_fp = os.path.join(wav_dp, f'{name}.wav')
    futures.append(executor.submit(partial(make_metadata, name, feats, wav_fp, out_dp)))
  # (name, prds, text, len_text, len_wav, len_spec, stats)
  metadata = [future.result() for future in tqdm(futures)]
  metadata = [mt for mt in metadata if mt is not None]

  # onely use sample within 2-sigma (95.45%) range of gauss distribution
  if DROPOUT_2SIGMA:
    tlens = np.asarray([mt[-4] for mt in metadata])   # mt[-4] := len_text
    tlens_mu, tlens_sigma = tlens.mean(), tlens.std()
    tlen_L = tlens_mu - 2 * tlens_sigma
    tlen_R = tlens_mu + 2 * tlens_sigma
    alens = np.asarray([mt[-2] for mt in metadata])   # mt[-2] := len_spec
    alens_mu, alens_sigma = alens.mean(), alens.std()
    alen_L = alens_mu - 2 * alens_sigma
    alen_R = alens_mu + 2 * alens_sigma

    metadata_filtered = []
    for mt in metadata:
      if not tlen_L <= mt[-4] <= tlen_R: continue
      if not alen_L <= mt[-2] <= alen_R: continue
      metadata_filtered.append(mt)
  else:
    metadata_filtered = metadata

  len_text   = np.asarray([mt[-4] for mt in metadata_filtered])
  len_wav    = np.asarray([mt[-3] for mt in metadata_filtered])
  len_spec   = np.asarray([mt[-2] for mt in metadata_filtered])
  stat_dicts = np.asarray([mt[-1] for mt in metadata_filtered])
  stats_agg = defaultdict(list)
  for stat in stat_dicts:
    for k, v in stat.items():
      stats_agg[k].append(v)
  stats_agg = { k: np.asarray(v) for k, v in stats_agg.items() }

  stats = {
    'total_examples': len(metadata_filtered),
    'total_hours':  len_wav.sum() / hp.sample_rate / (60 * 60),
    'min_len_txt':  len_text.min(),   # n_pinyins
    'max_len_txt':  len_text.max(),
    'avg_len_txt':  len_text.mean(),
    'min_len_wav':  len_wav.min(),    # n_samples
    'max_len_wav':  len_wav.max(),
    'avg_len_wav':  len_wav.mean(),
    'min_len_spec': len_spec.min(),   # n_frames
    'max_len_spec': len_spec.max(),
    'avg_len_spec': len_spec.mean(),
  }
  for k, v in stats_agg.items():
    try:
      agg_fn = k[:k.find('_')]
      stats[k] = getattr(v, agg_fn)()
    except:
      print(f'unknown aggregate method for {k}')

  # (name, prds, text)
  metadata = [mt[:3] for mt in metadata_filtered]
  return metadata, stats, wav_dp


def make_metadata(name, feats, wav_fp, out_dp):
  if not os.path.exists(wav_fp): return None
  text, prds = feats
  len_text = len(text.split(' '))
  if not len_text == len(prds): return None

  y = A.load_wav(wav_fp)
  y = A.trim_silence(y)
  y = A.align_wav(y)
  len_wav = len(y)

  y_cut = y[:-1]
  mag, mel = A.get_specs(y_cut)   # [M, T], [F, T]
  f0       = A.get_f0   (y_cut)   # [T,]
  c0       = A.get_c0   (y_cut)   # [T,]
  len_spec = mel.shape[1]

  assert len_wav == len_spec * hp.hop_length 

  np.save(os.path.join(out_dp, f'mel-{name}.npy'), mel, allow_pickle=False)
  np.save(os.path.join(out_dp, f'mag-{name}.npy'), mag, allow_pickle=False)
  np.save(os.path.join(out_dp, f'f0-{name}.npy'),  f0,  allow_pickle=False)
  np.save(os.path.join(out_dp, f'c0-{name}.npy'),  c0,  allow_pickle=False)

  stats = {
    'max_mel': mel.max(), 'min_mel': mel.min(), 
    'max_mag': mag.max(), 'min_mag': mag.min(), 
    'max_f0' : f0 .max(), 'min_f0' : f0 .min(), 
    'max_c0' : c0 .max(), 'min_c0' : c0 .min(), 
  }
  return (name, prds, text, len_text, len_wav, len_spec, stats)


def parse_label_file(fp) -> Dict[str, Tuple[str, str]]:
  '''prodosy:
       0: 词内部
       1: 连读的分词末
       2: 长音或停顿的分词末
       3: 分句末
       4: 句末
       5: 末尾标记
    韵律的层级：
      音节串#0 -> 单词串#1 -> 短语串#2 -> 短句串#3 -> 整句#4
  '''
  
  r = { }
  with open(fp, encoding='utf-8') as fh:
    while True:
      name_kanji = fh.readline().strip()
      if not name_kanji: break
      
      name, kanji = name_kanji.split('\t')      # '002333', '这是个#1例子#2' 
      pinyin = fh.readline().strip().lower()    # 'zhe4 shi4 ge4 li4 zi5'
      kanji = PUNCT_KANJI_REGEX.sub('', kanji)

      prodosy = []
      for k in kanji:
        if k == '#': continue
        if k.isdigit():
          if prodosy: prodosy[-1] = k
          else: prodosy.append(k)
        else: prodosy.append('0')
      prodosy = ''.join(prodosy)                # '00102'

      r[name] = (pinyin, prodosy)
  return r
