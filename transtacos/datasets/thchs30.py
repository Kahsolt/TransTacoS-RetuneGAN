import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

import audio

# NOTE: this is broken, do not use without modify!


def preprocess(args):
  in_dir = os.path.join(args.base_dir, 'thchs30')
  if not os.path.exists(in_dir):
    in_dir = os.path.join(args.base_dir, 'data_thchs30')
  out_dir = os.path.join(args.base_dir, args.out_dir)
  os.makedirs(out_dir, exist_ok=True)

  executor = ProcessPoolExecutor(max_workers=args.num_workers)
  futures = []
  dp = os.path.join(in_dir, 'data')
  for fn in (fn for fn in os.listdir(dp) if fn.endswith('.wav')):
    wav_path = os.path.join(dp, fn)
    with open(wav_path + '.trn', encoding='utf8') as fh:
      fh.readline()                 # ignore first line (kanji)
      text = fh.readline().strip()  # use pinyin only
    id = os.path.splitext(fn)[0]       # '<uid>_<sid>'
    futures.append(executor.submit(partial(_process_utterance, out_dir, id, wav_path, text)))
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, id, wav_path, text):
  wav = audio.load_wav(wav_path)
  wav = audio.trim_silence(wav)

  mag, mel = audio.get_specs(wav)
  mag = mag.astype(np.float32)
  mel = mel.astype(np.float32)
  n_frames = mel.shape[1]

  spec_fn = 'thchs30-spec-%s.npy' % id
  np.save(os.path.join(out_dir, spec_fn), mag.T, allow_pickle=False)
  mel_fn = 'thchs30-mel-%s.npy' % id
  np.save(os.path.join(out_dir, mel_fn), mel.T, allow_pickle=False)

  return (spec_fn, mel_fn, n_frames, text)
