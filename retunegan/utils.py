import os
import glob
from time import time

import torch
import matplotlib
import hparam as hp
if not hp.debug: matplotlib.use("Agg")
import matplotlib.pylab as plt

LRELU_SLOPE = 0.15
PI = 3.14159265358979


# plot
def plot_spectrogram(spec):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spec, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig


# network
def init_weights(m, mean=0.0, std=0.01):
    if 'Conv' in m.__class__.__name__:
        #m.weight.data.normal_(mean, std)
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu', a=LRELU_SLOPE)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def get_same_padding(kernel_size, dilation=1):
  return dilation * (kernel_size // 2)


def truncate_align(x, y):
    d = x.shape[-1] - y.shape[-1]
    if d != 0:
      print('[truncate_align] x.shape:', x.shape, 'y.shape:', y.shape)
      if   d > 0: x = x[:, :,   d //2 : -(  d  -   d //2)]
      elif d < 0: y = y[:, :, (-d)//2 : -((-d) - (-d)//2)]
    return x, y


def get_param_cnt(model):
    return sum(param.numel() for param in model.parameters())


def stat_grad(model, name):
    max_grad, min_grad = 0, 1e5
    for p in model.parameters():
        vabs = p.abs()
        vmin, vmax = vabs.min(), vabs.max()
        if vmin < min_grad: min_grad = vmin
        if vmax > max_grad: max_grad = vmax
    print(f'grad_{name}: max = {max_grad.item()}, min={min_grad.item()}')


# ckpt
def load_checkpoint(fp, device):
    assert os.path.isfile(fp)
    print(f"Loading '{fp}'")
    checkpoint_dict = torch.load(fp, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(fp, obj):
    print(f"Saving checkpoint to {fp}")
    torch.save(obj, fp)
    print("Complete.")


def scan_checkpoint(dp, prefix):
    pattern = os.path.join(dp, prefix + '*')
    cp_list = glob.glob(pattern)
    return len(cp_list) and sorted(cp_list)[-1] or None


# decorator
def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.2f}')
    return r
  return wrapper
