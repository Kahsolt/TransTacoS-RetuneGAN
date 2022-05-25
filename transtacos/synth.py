from time import time
import numpy as np
import tensorflow as tf

import hparam as hp
from models.tacotron import Tacotron
from text.text import text_to_phoneme, phoneme_to_sequence, sequence_to_phoneme
import audio as A
from text.symbols import _eos, _sep, get_vocab_size
from text.phonodict_cn import phonodict


class Synthesizer:
  
  def load(self, log_dir):
    print('Constructing tacotron model')

    # init data placeholder
    if hp.g2p == 'seq':    text_shape = [1, None]
    elif hp.g2p == 'syl4': text_shape = [1, None, 2]
    text_lengths = tf.placeholder(tf.int32, [1],        'text_lengths')   # bs=1 for one sample
    text         = tf.placeholder(tf.int32, text_shape, 'text')
    with tf.variable_scope('model'):
      self.model = Tacotron(hp)
      self.model.initialize(text_lengths, text)
      self.mag_output = self.model.mag_outputs[0]

    # load ckpt
    checkpoint_state = tf.train.get_checkpoint_state(log_dir)
    print('Resuming from checkpoint: %s' % checkpoint_state.model_checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_state.model_checkpoint_path)

  def synthesize(self, text, out_type='wav') -> bytes:
    # ref: `data.DataFeeder.load_data`
    if hp.g2p == 'seq':
      text = text + _eos
      print('text: ', text)
      phs = text_to_phoneme(text)
      print('phs: ', phs)
      seq = phoneme_to_sequence(phs)
      print('seq: ', seq)
      phs_rev = sequence_to_phoneme(seq)
      print('phs_rev: ', phs_rev)
    elif hp.g2p == 'syl4':
      C, V, T, Vx = text_to_phoneme(text)   # [[str]]

      CVVx, Tx = [ ], [ ]
      n_syllable = len(C)
      for i in range(n_syllable):
        if C[i] != phonodict.vacant:
          CVVx.append(C[i])  ; Tx.append(T[i])
        if V[i] != phonodict.vacant:
          CVVx.append(V[i])  ; Tx.append(T[i])
        if Vx[i] != phonodict.vacant:
          CVVx.append(Vx[i]) ; Tx.append(T[i])
        
        CVVx.append(_sep) ; Tx.append(0)
      
      # NOTE: pad <EOS> here, then convert to id_seq
      CVVx = phoneme_to_sequence(CVVx + [_eos])    # see phone table
      Tx = [int(t) for t in Tx] + [0]              # should be 0 ~ 5

      assert len(CVVx) == len(Tx)
      assert 0 <= min(CVVx) and max(CVVx) < get_vocab_size()
      assert 0 <= min(Tx)   and max(Tx)   < 6

      seq = np.stack([CVVx, Tx], axis=-1)    # [T, 2]
    
    seq = np.asarray(seq, dtype=np.int32)
    
    feed_dict = {
      self.model.text_lengths: [len(seq)],   # len(id_seq)
      self.model.text:         [seq],        # id_seq
    }
    mag = self.session.run(self.mag_output, feed_dict=feed_dict)
    mag = mag.T  # [F-1, T]
    if out_type == 'wav':
      wav = A.inv_spec(mag)                 # vocode with internal Griffin-Lim
      wav = A.trim_silence(wav)
      return wav                            # only data chunk, no RIFF capsulation
    if out_type == 'spec':
      S = A.spec_to_natural_scale(mag)      # denorm
      S = A.fix_zero_DC(S)
      return S
