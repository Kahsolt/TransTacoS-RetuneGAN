#!/usr/bin/env python3
# Author: Armit
# Create Time: 2021/3/29

from typing import List

from .symbols import _unk
from .phonodict_cn import phonodict


def to_syl4(pinyin:str, sep=' ') -> List[List[str]]:
  C, V, T, Vx = [], [], [], []

  py_ls = pinyin.split(sep)
  n_syllable = len(py_ls)
  for py in py_ls:
    # split tone
    t = py[-1]
    if t.isdigit(): py = py[:-1]
    else: t = '5'

    # deletec R-ending
    r_ending = False
    if py[-1] == 'r':
      r_ending = True
      if py != 'er':
        py = py[:-1]
    
    # split CV
    try:
      c, v, e = phonodict[py]
      C.append(c) ; V.append(v) ; T.append(t)
      if r_ending: Vx.append('_R')     # let R overriding N or NG
      else:        Vx.append(e)

    except:
      C.append(_unk)  ; V.append(_unk)  ; T.append(_unk) ; Vx.append(_unk)
      print('[Syllable] cannot parse %r' % py)

  assert len(C) == len(V) == len(T) == len(Vx) == n_syllable
  return [C, V, T, Vx]


def from_syl4(syl4:List[List[str]], sep=' ') -> str:
  return sep.join([''.join(s) for s in zip(*syl4)])


if __name__ == '__main__':
  pinyin = 'zi3 se4 de hua1 er2 wei4 shen2 me zher4 yang4 yuan2'
  print('pinyin:', pinyin)
  syl4 = to_syl4(pinyin)
  print('syl4:', syl4)
  syl4_serial = from_syl4(syl4)
  print('syl4_serial:', syl4_serial)
