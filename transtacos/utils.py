import atexit
from datetime import datetime

import matplotlib ; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import hparam as hp

_log_fmt = '%Y-%m-%d %H:%M:%S.%f'
_log_fp = None


def log_init(fp):
  global _log_fp
  _close_logfile()
  
  _log_fp = open(fp, 'a')
  _log_fp.write('\n')
  _log_fp.write('-----------------------------------------------------------------\n')
  _log_fp.write('  Starting new training run\n')
  _log_fp.write('-----------------------------------------------------------------\n')


def log(msg):
  print(msg)
  if _log_fp:
    _log_fp.write('[%s]  %s\n' % (datetime.now().strftime(_log_fmt)[:-3], msg))


def _close_logfile():
  global _log_fp
  if _log_fp:
    _log_fp.close()
    _log_fp = None


atexit.register(_close_logfile)


def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  plt.xlabel('Decoder timestep' + (f'\n\n{info}' if info else ''))
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.savefig(path, format='png')


def plot_specs(specs, path, info=None):
  # mel_g  mel_r
  # mag_g  mag_r
  ax = plt.subplot(221) ; sns.heatmap(specs[0]) ; ax.invert_yaxis()
  ax = plt.subplot(222) ; sns.heatmap(specs[2]) ; ax.invert_yaxis()
  ax = plt.subplot(223) ; sns.heatmap(specs[1]) ; ax.invert_yaxis()
  ax = plt.subplot(224) ; sns.heatmap(specs[3]) ; ax.invert_yaxis()
  plt.xlabel(info)
  plt.tight_layout()
  plt.margins(0, 0)
  plt.savefig(path, format='png', dpi=400)


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


class ValueWindow():    # NOTE: 右进左出的定长队列
  
  def __init__(self, window_size=100):
    self._window_size = window_size
    self._values = []

  def append(self, x):
    self._values = self._values[-(self._window_size - 1):] + [x]

  @property
  def sum(self):
    return sum(self._values)

  @property
  def count(self):
    return len(self._values)

  @property
  def average(self):
    return self.sum / max(1, self.count)

  def reset(self):
    self._values = []
