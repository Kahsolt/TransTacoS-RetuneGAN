import io
import pickle
from time import time
from argparse import ArgumentParser

import torch
import numpy as np
from flask import Flask, request, jsonify, send_file

import hparam as h
from models.generator import *
from audio import mag_to_mel, inv_mag
from utils import scan_checkpoint, load_checkpoint


os.environ['LIBROSA_CACHE_LEVEL'] = '50'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.enable    = True
torch.backends.cudnn.benchmark = True
torch.     manual_seed(h.randseed)
torch.cuda.manual_seed(h.randseed)

torch.autograd.set_detect_anomaly(hp.debug)


# globals
app = Flask(__name__)
generator = None


# vocode
@app.route('/vocode', methods=['POST'])
def vocode():
  try:
    # chk mag
    mag = pickle.loads(request.data)
    print(f'mag.shape: {mag.shape}, dyn_range: [{mag.min()}, {mag.max()}]')
    if mag.shape[1] == h.n_freq: mag = mag.T   # assure [F, T]
    # ref: preprocess in `data.Dataset.__getitem__()`
    mel      = mag_to_mel(mag)
    wavlen   = h.hop_length * mag.shape[1]
    wav_tmpl = inv_mag(mag, wavlen=wavlen-1)
    wav_tmpl = np.pad(wav_tmpl, (0, 1))

    # mel to wav
    s = time()
    with torch.no_grad():
      mel      = torch.from_numpy(mel)     .to(device, non_blocking=True).float().unsqueeze(0)
      wav_tmpl = torch.from_numpy(wav_tmpl).to(device, non_blocking=True).float().unsqueeze(0).unsqueeze(1)
      y_g_hat = generator(mel, wav_tmpl)
      wav = y_g_hat.squeeze()
      wav = wav.cpu().numpy().astype(np.float32)
    t = time()
    print(f'wav.shape: {wav.shape}, dyn_range: [{wav.min()}, {wav.max()}]')
    print(f'[Vocode] Done in {t - s:.2f}s')

    # transfer
    bio = io.BytesIO()
    bio.write(pickle.dumps(wav))  # float32 -> byte
    bio.seek(0)     # reset fp to beginning for `send_file` to read
    return send_file(bio, mimetype='application/octet-stream')

  except Exception as e:
    print('[Error] %r' % e)
    return jsonify({'error': e})


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--log_path', required=True)
  parser.add_argument('--host', type=str, default='0.0.0.0')
  parser.add_argument('--port', type=int, default=5104)
  args = parser.parse_args()

  # load ckpt
  generator = globals().get(f'Generator_{h.generator_ver}')().to(device)
  state_dict_g = load_checkpoint(scan_checkpoint(args.log_path, 'g_'), device)
  generator.load_state_dict(state_dict_g['generator'])
  generator.eval()
  generator.remove_weight_norm()

  app.run(host=args.host, port=args.port, debug=False)
