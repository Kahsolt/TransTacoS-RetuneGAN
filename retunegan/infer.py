import os
import sys
import importlib
from argparse import ArgumentParser
from retunegan.audio import get_mag, inv_mag

import torch
import numpy as np

from models import Generator
from audio import load_wav, save_wav, get_mel
from utils import load_checkpoint, scan_checkpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = None


def load_generator(a):
    global generator
    if not generator:
        Generator = globals().get(f'Generator_{h.generator_ver}')
        generator = Generator().to(device)
        state_dict_g = state_dict_g = load_checkpoint(scan_checkpoint(args.log_path, 'g_'), device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
    return generator


def inference(a, x, wav_ref):
    generator = load_generator(a)
    with torch.no_grad():
        y_g_hat = generator(x, wav_ref)
        wav = y_g_hat.squeeze()
        wav = wav.cpu().numpy().astype(np.float32)
        return wav


def inference_from_mag(a, fp):
    x = np.load(fp)
    x = torch.from_numpy(x).to(device)
    if x.size(1) == h.n_freq: x = x.T

    if len(x.shape) < 3: x = x.unsqueeze(0)     # set batch_size=1
    y = inv_mag(x)
    wav = inference(a, x, y)

    wav_fp = os.path.join(a.output_dir, os.path.splitext(os.path.basename(fp))[0] + '_gen_from_mag.wav')
    save_wav(wav, wav_fp)
    print(f'   Done {wav_fp!r}')


def inference_from_wav(a, fp):
    wav = load_wav(fp)
    wav = torch.from_numpy(wav).to(device)
    x = get_mag(wav)

    if len(x.shape) < 3: x = x.unsqueeze(0)     # set batch_size=1
    y = inv_mag(x)
    wav = inference(a, x, y)

    wav_fp = os.path.join(a.input_path, os.path.splitext(os.path.basename(fp))[0] + '_gen_from_wav.wav')
    save_wav(wav, wav_fp)
    print(f'   Done {wav_fp!r}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_path', default='test')
    parser.add_argument('--log_path', required=True)
    a = parser.parse_args()

    # load frozen hparam
    sys.path.insert(0, a.log_path)
    h = importlib.import_module('hparam')
    torch.manual_seed(h.randseed)
    torch.cuda.manual_seed(h.randseed)

    print('Initializing Reference Process..')
    fps = [os.path.join(a.input_path, fn) for fn in os.listdir(a.input_path)]
    for fp in [fp for fp in fps if fp.lower().endswith('.npy')]:
        inference_from_mag(a, fp)
    for fp in [fp for fp in fps if fp.lower().endswith('.wav')]:
        inference_from_wav(a, fp)
