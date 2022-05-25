import os
import random
from threading import Thread
import traceback

import numpy as np
import tensorflow as tf

import hparam as hp
from text.text import text_to_phoneme, phoneme_to_sequence
from text.symbols import _eos, _sep, get_vocab_size
from text.phonodict_cn import phonodict
from audio import quantilize_c0, quantilize_f0


_batches_per_group = hp.batch_size
_pad = 0                  # FIXME: hardcoded for <PAD>/'_' fortext-token


class DataFeeder(Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, metadata_fp, hparams):
    super(DataFeeder, self).__init__()
    self._hparams = hparams
    self._session = None
    self._coord = coordinator
    self._offset = 0

    # Load metadata & make cache:
    self._datadir = os.path.dirname(metadata_fp)
    with open(metadata_fp, encoding='utf-8') as f:
      self._metadata = [line.strip().split('|') for line in f]
    self.data = [None] * len(self._metadata)

    # Create placeholders for inputs and targets. Don't specify batch size because we want to
    # be able to feed different sized batches at eval time.
    if hp.g2p == 'seq':
      text_shape = [None, None]
    elif hp.g2p == 'syl4':
      text_shape = [None, None, 2]
    self._placeholders = [
      tf.placeholder(tf.int32,   [None],                         'text_lengths'),
      tf.placeholder(tf.int32,   text_shape,                     'text'),
      tf.placeholder(tf.int32,   [None, None],                   'prds'),
      tf.placeholder(tf.int32,   [None],                         'spec_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.n_mel],    'mel_targets'),
      tf.placeholder(tf.float32, [None, None, hparams.n_freq-1], 'mag_targets'),
      tf.placeholder(tf.int32,   [None, None],                   'f0_targets'),
      tf.placeholder(tf.int32,   [None, None],                   'c0_targets'),
      tf.placeholder(tf.float32, [None, None],                   'stop_token_targets')
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(hp.batch_size, [h.dtype for h in self._placeholders], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    holders = queue.dequeue()
    for i, holder in enumerate(holders): holder.set_shape(self._placeholders[i].shape)
    (self.text_lengths, self.text, self.prds, self.spec_lengths, self.mel_targets, self.mag_targets, self.f0_targets, self.c0_targets, self.stop_token_targets) = holders

  def start_in_session(self, session):
    self._session = session
    self.start()

  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)

  def _enqueue_next_group(self):
    def _get_next_example():
      '''Loads a single example (input, mel_target, mag_target, stop_token_target, len(spec)) from memory cached'''

      if self._offset >= len(self.data):  # infinit loop
        self._offset = 0
        random.shuffle(self.data)

      if self.data[self._offset] is None:
        self.load_data(self._offset)
      data = self.data[self._offset]
      self._offset += 1
      return data

    # Read a group of examples:
    n = self._hparams.batch_size
    r = self._hparams.outputs_per_step
    examples = [_get_next_example() for _ in range(n * _batches_per_group)]    # group = batch_size * batches_per_group

    # Bucket examples based on similar output sequence length (spec n_frames) for efficiency
    # NOTE: 按照输出的帧长度排序，而不是输入的文本长度！
    examples.sort(key=lambda x: len(x[-1]))
    batches = [examples[i:i+n] for i in range(0, len(examples), n)]     # split to batches
    random.shuffle(batches)

    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)

  def load_data(self, index):
    '''Loads all examples [(input, mel_target, mag_target, stop_token_target, len(spec))] from disk'''
    
    meta = self._metadata[index]      # meta: (id, len(spec), text)
    id, prds, text = meta
    if hp.g2p == 'seq':
      # NOTE: pad <EOS> here, then convert to id_seq
      seq = phoneme_to_sequence(text_to_phoneme(text + _eos))
      prds = [int(d) for d in prds]
    elif hp.g2p == 'syl4':
      C, V, T, Vx = text_to_phoneme(text)   # [[str]]
      prds = [int(d) for d in prds]
      try:
        assert len(C) == len(prds)
      except:
        breakpoint()

      CVVx, Tx, P = [ ], [ ], [ ]
      n_syllable = len(C)
      for i in range(n_syllable):
        if C[i] != phonodict.vacant:
          CVVx.append(C[i])  ; Tx.append(T[i]) ; P.append(0)
        if V[i] != phonodict.vacant:
          CVVx.append(V[i])  ; Tx.append(T[i]) ; P.append(0)
        if Vx[i] != phonodict.vacant:
          CVVx.append(Vx[i]) ; Tx.append(T[i]) ; P.append(0)
        
        CVVx.append(_sep) ; Tx.append(0) ; P.append(prds[i])
      
      # NOTE: pad <EOS> here, then convert to id_seq
      CVVx = phoneme_to_sequence(CVVx + [_eos])    # see phone table
      Tx = [int(t) for t in Tx] + [0]              # should be 0 ~ 5
      for i in range(len(P) - 2, -1, -1):
        if P[i] == 0:
          P[i] = P[i + 1]
      P = P + [5]                                  # should be 0 ~ 5

      try:
        assert len(CVVx) == len(Tx) == len(P)
        assert 0 <= min(CVVx) and max(CVVx) < get_vocab_size()
        assert 0 <= min(P)    and max(P)    < hp.n_prds
        assert 0 <= min(Tx)   and max(Tx)   < hp.n_tone
      except:
        breakpoint()
      
      seq = np.stack([CVVx, Tx], axis=-1)    # [T, 2]
      prds = P
    else: raise

    text       = np.asarray(seq,  dtype=np.int32)
    prds       = np.asarray(prds, dtype=np.int32)
    mel_target = np.load(os.path.join(self._datadir, f'mel-{id}.npy')).T  # [T, F]
    mag_target = np.load(os.path.join(self._datadir, f'mag-{id}.npy')).T  # [T, M]
    f0_target  = np.load(os.path.join(self._datadir, f'f0-{id}.npy'))
    c0_target  = np.load(os.path.join(self._datadir, f'c0-{id}.npy'))
    stop_token_target = np.zeros(mel_target.shape[0])       # NOTE: 在有数据的mel帧上，初始化停止概率为0，另参见`_pad_stop_token_target()`

    mag_target = mag_target[:, 1:]   # remove DC
    f0_target = quantilize_f0(f0_target)
    c0_target = quantilize_c0(c0_target)
    try:
      assert 0 <= min(f0_target) and max(f0_target) < hp.n_f0_bins
      assert 0 <= min(c0_target) and max(c0_target) < hp.n_c0_bins
    except:
        breakpoint()

    #breakpoint()

    self.data[index] = (text, prds, mel_target, mag_target, f0_target, c0_target, stop_token_target)
  
def _prepare_batch(batch, outputs_per_step):   # FIXME: what for `outputs_per_step`
  # batch of data, one line per sample (input/id_seq, mel, mag, stop_token, len(spec))
  random.shuffle(batch)
  # pad within batch
  text_lengths       = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
  if hp.g2p == 'seq':    text = _prepare_inputs   ([x[0] for x in batch])
  elif hp.g2p == 'syl4': text = _prepare_inputs_2d([x[0] for x in batch])
  else: raise
  prds               = _prepare_inputs([x[1] for x in batch])
  spec_lengths       = np.asarray([len(x[2]) for x in batch], dtype=np.int32)
  mel_targets        = _prepare_targets([x[2] for x in batch], outputs_per_step)
  mag_targets        = _prepare_targets([x[3] for x in batch], outputs_per_step)
  f0_targets         = _prepare_stop_token_targets([x[4] for x in batch], outputs_per_step, 0)
  c0_targets         = _prepare_stop_token_targets([x[5] for x in batch], outputs_per_step, 0)
  stop_token_targets = _prepare_stop_token_targets([x[6] for x in batch], outputs_per_step, 1.0)
  return (text_lengths, text, prds, spec_lengths, mel_targets, mag_targets, f0_targets, c0_targets, stop_token_targets)


def _prepare_inputs(inputs):
  def _pad_input(x:list, length):                # pad <PAD>=0 for text
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

  max_len = max((len(x) for x in inputs))        # 填充到该batch中最长的长度，seq在前面已经附加了<EOS>
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_inputs_2d(inputs):
  def _pad_input(x:list, length):                # pad <PAD>=0 for text
    return np.pad(x, [(0, length - x.shape[0]), (0, 0)], mode='constant', constant_values=_pad)

  max_len = max((len(x) for x in inputs))        # 填充到该batch中最长的长度，seq在前面已经附加了<EOS>
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, r):
  def _pad_target(x, length):                    # pad <PAD>=0.0 for spec
    return np.pad(x, [(0, length - x.shape[0]), (0, 0)], mode='constant', constant_values=x.min())

  max_len = max((len(t) for t in targets)) + 1   # +1 for <EOS>
  max_len = _round_up(max_len, r)                # 上取整到r的整数倍
  return np.stack([_pad_target(x, max_len) for x in targets])


def _prepare_stop_token_targets(targets, r, pad_val):
  def _pad_stop_token_target(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=pad_val)   # NOTE: 对于填充的尾帧，初始化停止概率为1

  max_len = max((len(t) for t in targets)) + 1   # +1 for <EOS>
  max_len = _round_up(max_len, r)                # 上取整到r的整数倍
  return np.stack([_pad_stop_token_target(x, max_len) for x in targets])


def _round_up(x, multiple):                      # 向上捨入到multiple的整數倍
  remainder = x % multiple
  return x if remainder == 0 else (x + multiple - remainder)
