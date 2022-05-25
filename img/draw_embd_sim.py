import os
import sys

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def process(e, name):
  def dot_sim(x):
    return np.dot(x, x.T)

  def cosine_sim(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return dot_sim(x) / (np.dot(n, n.T) + 1e-8)

  #s1 = dot_sim(e)
  #np.save(os.path.join(BASE_PATH, f'{name}_dot.npy'), s1)
  #sns.heatmap(s1)
  #plt.gca().invert_yaxis()
  #plt.savefig(os.path.join(BASE_PATH, f'{name}_dot.png'))
  #plt.clf()
  
  s2 = cosine_sim(e)
  #np.save(os.path.join(BASE_PATH, f'{name}_cosine.npy'), s2)
  sns.heatmap(s2)
  plt.gca().invert_yaxis()
  plt.savefig(os.path.join(BASE_PATH, f'{name}_cosine.png'))
  plt.clf()


fp = sys.argv[1]
fn = os.path.basename(fp)
base, ext = os.path.splitext(fn)

d = np.load(fp, allow_pickle=True)
if isinstance(d, np.ndarray):
  process(d, base)
elif isinstance(d, dict):
  for k in d.keys():
    if not k.endswith('embd'): continue
    process(d[k], f'{base}_{k}')
else: raise
