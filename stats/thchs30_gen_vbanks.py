#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020/11/14 

# 将thchs30按音色分库，产生几个train.txt

import re
from pathlib import Path
from os import chdir as cd, getcwd as pwd
from collections import defaultdict

SPEC_DIR = 'thchs30.spec'
INDEX_FILE = 'train.txt'
BASE_PATH = Path(__file__).parent.absolute() / SPEC_DIR
R = re.compile(r'-([ABCD]\d+)_')

MALE_LIST = [ 'A8', 'B8', 'C8', 'D8' ]
FEMALE_POWER_LIST = [ 'A2', 'A4', 'A6', 'A14', 'A22', 'A34', 'B4', 'B6', 'B12', 'B22', 'B31', 'C4', 'C6', 'C31', 'D6', 'D31', 'D32' ]
FEMALE_SOFT_LIST = [ 'A7', 'A11', 'A19', 'B7', 'C7', 'C14', 'C17', 'C18', 'C20', 'C32', 'D7', 'D11' ]
CHILD_LIST = [ 'A13', 'B11', 'C12', 'C13', 'C19', 'C21', 'C22', 'D21' ]

# => {'uid': ['sample_config1', ...]}
def read_index(fn=INDEX_FILE) -> defaultdict:
  index_dict = defaultdict(list)
  with open(INDEX_FILE) as fh:
    samples = fh.read().split('\n')
  for s in samples:
    if len(s) == 0: continue
    spec_fn, mel_fn, _, txt = s.split('|')
    id = R.findall(spec_fn)[0]
    index_dict[id].append(s)
  return index_dict

def write_index(fn, vbank):
  with open(fn, 'w') as fh:
    for s in vbank:
      fh.write(s)
      fh.write('\n')

# => ['sample_config1', ...]
def gen_index(index, vt) -> list:
  uid_list = globals().get(vt.upper() + '_LIST', [])
  vbank = [ ]
  for uid in uid_list:
    vbank.extend(index[uid])
  return vbank

if __name__ == '__main__':
  savedp = pwd()
  cd(BASE_PATH)
  index = read_index()
  for vt in ['male', 'female_power', 'female_soft', 'child']:
    vbank = gen_index(index, vt)
    write_index('vbank_' + vt + '.txt', vbank)
  cd(savedp)
