#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/01/07 

import os
import random
from pprint import pformat
from argparse import ArgumentParser
from importlib import import_module
from typing import List, Tuple

import hparam as hp
random.seed(hp.randseed)


def write_metadata(metadata:Tuple[List, List], stats:dict, wav_path, args):
  if args.shuffle: random.shuffle(metadata)
  
  out_path = os.path.join(args.base_dir, args.out_dir)
  os.makedirs(out_path, exist_ok=True)

  cp = int(len(metadata) * args.split_ratio)
  mt_test, mt_train = metadata[:cp], metadata[cp:]
  
  with open(os.path.join(out_path, 'train.txt'), 'w', encoding='utf-8') as fh:
    for mt in mt_train:
      fh.write('|'.join([str(x) for x in mt]))
      fh.write('\n')

  with open(os.path.join(out_path, 'test.txt'), 'w', encoding='utf-8') as fh:
    for mt in mt_test:
      fh.write('|'.join([str(x) for x in mt]))
      fh.write('\n')

  with open(os.path.join(out_path, 'stats.txt'), 'w', encoding='utf-8') as fh:
    for k, v in stats.items():
      fh.write(f'{k}\t{v}')
      fh.write('\n')

  with open(os.path.join(out_path, 'wav_path.txt'), 'w', encoding='utf-8') as fh:
    fh.write(wav_path)


if __name__ == '__main__':
  def str2bool(s:str) -> bool:
    s = s.lower()
    if s in ['true',  't', '1']: return True
    if s in ['false', 'f', '0']: return False
    raise ValueError(f'invalid bool value: {s}')
  
  base_dir = os.path.dirname(os.path.abspath(__file__))
  DATASETS = [fn[:-3] for fn in os.listdir(os.path.join(base_dir, 'datasets')) if not fn.startswith('__')]

  parser = ArgumentParser()
  parser.add_argument('--base_dir',   required=True,                         help='base path containing the dataset folder')
  parser.add_argument('--out_dir',                    default='preprocessed', help='preprocessed output folder')
  parser.add_argument('--dataset',     required=True, choices=DATASETS)
  parser.add_argument('--shuffle',     type=str2bool, default=True,           help='shuffle metadata')
  parser.add_argument('--split_ratio', type=float,    default=0.05,           help='test/train split')
  parser.add_argument('--num_workers', type=int,      default=4)
  args = parser.parse_args()

  os.environ['LIBROSA_CACHE_LEVEL'] = '50'

  proc = import_module(f'datasets.{args.dataset}')
  metadata, stats, wav_path = proc.preprocess(args)
  print('wav_path:', wav_path)
  print('stats:', pformat(stats))    # why `sort_dicts=False` not work
  write_metadata(metadata, stats, wav_path, args)
