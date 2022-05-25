#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/04/15 

import os
import re
import pickle
from time import time
from argparse import ArgumentParser
from tempfile import gettempdir

import numpy as np
from flask import Flask, request, jsonify, send_file
from requests.utils import unquote
from scipy.io import wavfile
from xpinyin import Pinyin
from requests import session


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(BASE_PATH, 'index.html')

TMP_DIR = gettempdir()
WAV_TMP_FILE = os.path.join(TMP_DIR, 'synth.wav')
MP3_TMP_FILE = os.path.join(TMP_DIR, 'synth.mp3')

REGEX_PUNCT_IGNORE = re.compile('、|：|；|“|”|‘|’')
REGEX_PUNCT_BREAK  = re.compile('，|。|！|？')
MAX_CLUASE_LENGTH  = 20
CONVERT_MP3 = False

SAMPLE_RATE = 22050
SYNTH_API   = 'http://127.0.0.1:5105/synth_spec'
VOCODER_API = 'http://127.0.0.1:5104/vocode'


app = Flask(__name__)
html_page = None
http = session()
kanji2pinyin = Pinyin()


def synth_and_save_file(txt):
  # Text-Norm
  if True:
    s = time()
    print(f'text/raw: {txt!r}')

    kanji = REGEX_PUNCT_IGNORE.sub('', txt)
    kanji = REGEX_PUNCT_BREAK.sub(' ', txt)
    segs = ['']   # dummy init
    for rs in [s.strip() for s in kanji.split(' ') if s.strip()]:
      if (not segs[-1]) or (len(rs) + len(segs[-1]) < MAX_CLUASE_LENGTH):
        segs[-1] = segs[-1] + rs
      else: segs.append(rs)
    print(f'text/segs: {segs!r}')
    t = time()
    print('[TextNorm] Done in %.2fs' % (t - s))
  
  # Synth
  if True:
    s = time()
    spec_clips = []
    for seg in segs:
      pinyin = ' '.join(kanji2pinyin.get_pinyin(seg, tone_marks='numbers').split('-'))
      resp = http.post(SYNTH_API, json={'pinyin': pinyin})
      spec = pickle.loads(resp.content)
      spec_clips.append(spec)
    spec = np.concatenate(spec_clips)
    print('spec.shape:', spec.shape)
    t = time()
    print('[Synth] Done in %.2fs' % (t - s))

  # Vocode
  if True:
    s = time()
    resp = http.post(VOCODER_API, data=pickle.dumps(spec))
    wav = pickle.loads(resp.content)
    wavfile.write(WAV_TMP_FILE, SAMPLE_RATE, wav)
    print('wav.length:', len(wav))
    t = time()
    print('[Vocode] Done in %.2fs' % (t - s))

  # Compress
  if CONVERT_MP3:
    s = time()
    cmd = f'ffmpeg -i "{WAV_TMP_FILE}" -f mp3 -acodec libmp3lame -y "{MP3_TMP_FILE}" -loglevel quiet'
    r = os.system(cmd)
    t = time()
    print('[Compress] Done in %.2fs' % (t - s))


@app.route('/', methods=['GET'])
def root():
  global html_page
  if not html_page:
    with open(HTML_FILE, encoding='utf-8') as fp:
      html_page = fp.read()
  return html_page


@app.route('/synth', methods=['GET'])
def synth():
  txt = unquote(request.args.get('text')).strip()
  if not txt: return jsonify({'error': 'empty request'})

  try:
    synth_and_save_file(txt)
    if CONVERT_MP3:
      return send_file(MP3_TMP_FILE, mimetype='audio/mp3')
    else:
      return send_file(WAV_TMP_FILE, mimetype='audio/wav')
  except Exception as e:
    print('[Error] %r' % e)
    return jsonify({'error': e})


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--host', type=str, default='0.0.0.0')
  parser.add_argument('--port', type=int, default=5103)
  args = parser.parse_args()

  app.run(host=args.host, port=args.port, debug=False)
