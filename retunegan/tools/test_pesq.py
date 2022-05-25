#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/03/10

# use this script evaluate PESQ

import sys
import librosa as L
from pesq import pesq

sr = 16000
#sr = 8000

BASE_PATH = r'C:\Users\Kahsolt\Desktop\Workspace\Essay\基于韵律优化与波形修复的汉语语音合成方法研究\audio'


def test(fp_y, fp_y_hat):
  ref, _ = L.load(BASE_PATH + '\\' + fp_y,     sr)
  deg, _ = L.load(BASE_PATH + '\\' + fp_y_hat, sr)

  print(pesq(sr, ref, ref, 'wb'))
  print(pesq(sr, ref, deg, 'wb'))

print('[gl]')
test('gl_gt.wav', 'gl_64i.wav')
print('[mlg]')
test('mlg-gt.wav', 'mlg-100e.wav')
print('[hfg-40k]')
test('hfg-gt.wav', 'hfg-40k.wav')
print('[hfg-85k]')
test('hfg-gt.wav', 'hfg-85k.wav')

print('[taco]')
test('taco-gt.wav', 'taco-103k.wav')
