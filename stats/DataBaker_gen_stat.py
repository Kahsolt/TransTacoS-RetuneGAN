#!/usr/bin/env python3
# Author: Armit
# Create Time: 2021/01/07 

import tgt
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from os import chdir as cd, getcwd as pwd, listdir

BASE_PATH = Path(__file__).parent.absolute()
WORKING_DIR = BASE_PATH / 'DataBaker.preproc' / 'TextGrid'
STAT_OUT_FILE_FMT = BASE_PATH / 'DataBaker.stat-%s.csv'

def collect_stat(by_name='phones'):
  durdict = defaultdict(list)
  for fn in listdir():
    tg = tgt.read_textgrid(fn)
    for ph in tg.get_tier_by_name(by_name).intervals:
      durdict[ph.text].append(ph.duration())
  
  stat = {k: (len(v), np.mean(v), np.std(v), np.min(v), np.max(v)) for k, v in durdict.items()}
  df = pd.DataFrame(stat, index=['freq', 'mean', 'std', 'min', 'max']).T
  df.to_csv(str(STAT_OUT_FILE_FMT) % by_name)

if __name__ == '__main__':
  savedp = pwd()
  cd(WORKING_DIR)
  for name in ['words', 'phones']:
    collect_stat(name)
  cd(savedp)
