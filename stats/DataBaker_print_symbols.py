#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020/12/16 

# 收集DataBaker中出现过的拼音音节，打印其列表

import re
from math import ceil
from pathlib import Path
from os import chdir as cd, getcwd as pwd

SPEC_DIR = 'DataBaker.preproc'
BASE_PATH = Path(__file__).parent.absolute() / SPEC_DIR
INDEX_FILES = ['train.txt', 'val.txt']

def collect_symbols():
  symbols = set()
  for fn in INDEX_FILES:
    with open(fn) as fh:
      samples = fh.read().split('\n')
    for s in samples:
      if len(s) == 0: continue
      _, txt = s.split('|')
      symbols = symbols.union(txt[1:-1].split(' '))
  return sorted(list(symbols))

def pprint_symbols(symbols):
  n_sym = len(symbols)
  SYM_PER_LINE = 15
  n_line = ceil(n_sym / SYM_PER_LINE)
  
  print('_pinyin = [', end='')
  for idx, sym in enumerate(symbols):
    col = idx % SYM_PER_LINE
    if col == 0: print('\n  ', end='')
    print(f"'{sym}', ", end='')
  print(']')

if __name__ == '__main__':
  savedp = pwd()
  cd(BASE_PATH)
  pprint_symbols(collect_symbols())
  cd(savedp)
