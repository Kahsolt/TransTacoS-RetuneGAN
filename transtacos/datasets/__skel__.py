#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/01/10 

# write your own dataset preprocesser
# here's the template :)

import os
from typing import List, Tuple


def preprocess(args) -> Tuple[List[Tuple], dict, str]:
  # wav_dp is a file path string, pointing to the folder containing *.wav files
  wav_dp = os.path.join(args.base_path, 'dataset', 'wavs')

  # metadata is a list containing textual informatin  
  metadata = [
    # for exmaple, name-text pairs
    ('00001', 'this is an exmaple'),
    ('00002', 'this is another exmaple'),
    ('00003', 'yet another exmaple'),
  ]

  # stats is a dictionary about statistcs  
  stats = {
    'min_len_txt': 18,
    'max_len_txt': 23,
    'avg_len_txt': 20.0,
    'min_len_wav': 100,
    'max_len_wav': 200,
    'avg_len_wav': 150.0,
  }

  # return them all
  return metadata, stats, wav_dp
