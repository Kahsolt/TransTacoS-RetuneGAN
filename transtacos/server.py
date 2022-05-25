import os
import io
import pickle
from re import compile as Regex
from time import time
from tempfile import gettempdir
from argparse import ArgumentParser

import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, send_file
from requests.utils import unquote
from xpinyin import Pinyin

import hparam as hp
from synth import Synthesizer
from audio import save_wav 


for gpu in tf.config.experimental.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_XLA_FLAGS']         = '--tf_xla_cpu_global_jit'
os.environ['XLA_FLAGS']            = '--xla_hlo_profile'

np.random.seed           (hp.randseed)
tf.random.set_random_seed(hp.randseed)


# globals
app = Flask(__name__)
kanji2pinyin = Pinyin()
synthesizer = None
html_text = None

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(BASE_PATH, '..', 'index.html')

TMP_DIR = gettempdir()
WAV_TMP_FILE = os.path.join(TMP_DIR, 'synth.wav')

REGEX_PUNCT_IGNORE = Regex('、|：|；|“|”|‘|’')
REGEX_PUNCT_BREAK  = Regex('，|。|！|？')
MAX_CLUASE_LENGTH  = 20


# quick demo index page
@app.route('/', methods=['GET'])
def root():
  global html_text
  if not html_text:
    with open(HTML_FILE, encoding='utf-8') as fp:
      html_text = fp.read()
  return html_text


# vocode with internal Griffin-Lim
@app.route('/synth', methods=['GET'])
def synth():
  kanji = unquote(request.args.get('text'))
  
  if kanji:
    try:
      # Text-Norm
      if True:
        s = time()
        print(f'text/raw: {kanji!r}')

        kanji = REGEX_PUNCT_IGNORE.sub('', kanji)
        kanji = REGEX_PUNCT_BREAK.sub(' ', kanji)
        segs = ['']   # dummy init
        for rs in [s.strip() for s in kanji.split(' ') if s.strip()]:
          if (not segs[-1]) or (len(rs) + len(segs[-1]) < MAX_CLUASE_LENGTH):
            segs[-1] = segs[-1] + rs
          else: segs.append(rs)
        print(f'text/segs: {segs!r}')
        t = time()
        print(f'[TextNorm] Done in {t - s:.2f}s')
      
      # Synth
      if True:
        s = time()
        wav_clips = []
        for seg in segs:
          text = ' '.join(kanji2pinyin.get_pinyin(seg, tone_marks='numbers').split('-'))
          wav = synthesizer.synthesize(text, 'wav')
          wav_clips.append(wav)
        wav = np.concatenate(wav_clips)
        print('wav.shape:', wav.shape)
        t = time()
        print(f'[Synth] Done in {t - s:.2f}s')
      
      # Save file
      if True:
        s = time()
        save_wav(wav, WAV_TMP_FILE)
        t = time()
        print(f'[SaveFile] Done in {t - s:.2f}s')

      return send_file(WAV_TMP_FILE, mimetype='audio/wav')
    except Exception as e:
      print('[Error] %r' % e)
      error_msg = 'synth failed, see logs'
  else:
    error_msg = 'bad request params or no text to synth?'
  
  return jsonify({'error': error_msg})


# return linear spec
@app.route('/synth_spec', methods=['POST'])
def synth_spec():
  try:
    # chk txt
    pinyin = request.get_json().get('pinyin').strip()
    if not pinyin:
      return jsonify({'error': 'no text to synth'})

    # text to mag
    s = time()
    spec = synthesizer.synthesize(pinyin, 'spec')
    print('spec.shape:', spec.shape)
    t = time()
    print(f'[Synth] Done in {t - s:.2f}s')

    # transfer
    bio = io.BytesIO()
    bio.write(pickle.dumps(spec))  # float32 -> byte
    bio.seek(0)     # reset fp to beginning for `send_file` to read
    return send_file(bio, mimetype='application/octet-stream')
  
  except Exception as e:
    print('[Error] %r' % e)
    return jsonify({'error': e})


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--log_path', required=True)
  parser.add_argument('--host', type=str, default='0.0.0.0')
  parser.add_argument('--port', type=int, default=5105)
  args = parser.parse_args()

  # load ckpt
  synthesizer = Synthesizer()
  synthesizer.load(args.log_path)

  app.run(host=args.host, port=args.port, debug=False)
