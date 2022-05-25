import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.keras.backend import batch_dot

import hparam as hp

REUSE = False


''' below is for tacotron compactible '''

def prenet(inputs, layer_sizes, is_training, scope='prenet'):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0   # NOTE: only dropout on trainning
  with tf.variable_scope(scope):
    # NOTE: chained i-layers of dense and dropout
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
  return x


def conv1d(inputs, k, filters, activation, is_training, scope='conv1d'):
  with tf.variable_scope(scope):
    conv = tf.layers.conv1d(
      inputs,
      filters=filters,
      kernel_size=k,
      activation=None,
      padding='same')
    bn = tf.layers.batch_normalization(conv, training=is_training)
    return activation(bn)


def highwaynet(inputs, depth, scope='highwaynet'):
  with tf.variable_scope(scope):
    H = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.relu,
      name='H')
    T = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.sigmoid,
      name='T',
      bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def cbhg(inputs, input_lengths, K, proj_dims, depth, is_training, scope='cbhg'):
  with tf.variable_scope(scope):
    with tf.variable_scope('conv_bank'):
      # Convolution bank: concatenate on the last axis to connect channels from all convolutions
      conv = tf.concat(
        [conv1d(inputs, k+1, depth//2, tf.nn.relu, is_training, 'conv1d_%d' % (k+1)) for k in range(K)],
        axis=-1)

    # Maxpooling:
    conv = tf.layers.max_pooling1d(
      conv,
      pool_size=2,
      strides=1,
      padding='same')

    # Two projection layers: reduce depth
    proj = conv1d(conv, 3, proj_dims[0], tf.nn.relu, is_training, 'proj_1')  # depth: 896(7*128) -> 128
    proj = conv1d(proj, 3, proj_dims[1], lambda _:_, is_training, 'proj_2')     # depth: 128 -> 256

    # Residual connection:
    # now we marge `acoustics (phoneme)` with `pardosy (text context)`
    highway_input = inputs + proj

    # Handle dimensionality mismatch:
    if highway_input.shape[-1] != depth:
      highway_input = tf.layers.dense(highway_input, depth)
    # 4-layer HighwayNet:
    for i in range(hp.highway_layers):
      highway_input = highwaynet(highway_input, depth,  'highway_%d' % (i+1))

    # Bidirectional RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      GRUCell(depth//2),
      GRUCell(depth//2),
      highway_input,
      sequence_length=input_lengths,
      dtype=tf.float32)
    
    return tf.concat(outputs, axis=-1)   # Concat forward and backward


''' below is my stuff '''

def gaussian_noise(x, is_training):
  if hp.hidden_gauss_std:
    x = tf.keras.layers.GaussianNoise(hp.hidden_gauss_std)(x, training=is_training)
  return x


def conv_stack(x, n_layers, k, d_in, d_out, activation=tf.nn.relu, scope='conv_stack'):
  with tf.variable_scope(scope, reuse=REUSE):
    for i in range(n_layers-1):
      x = tf.layers.conv1d(x, d_in, k, padding='same', name=f'conv{i+1}')
      x = activation(x)
    x = tf.layers.conv1d(x, d_out, k, padding='same', name=f'conv{n_layers}')
    return x


def dot_attn(x, y, mask, attn_dim, scope='dot_attn'):
  with tf.variable_scope(scope, reuse=REUSE):
    # [B, N, A]
    q = tf.layers.dense(x, attn_dim, name='q')
    # [B, T, A]
    k = tf.layers.dense(y, attn_dim, name='k')
    v = tf.layers.dense(y, attn_dim, name='v')

    # [B, N, T]
    e = tf.matmul(q, k, transpose_b=True)
    e = e * mask + (1 - mask) * -1e8        # mask energe to inf
    e = e / tf.sqrt(tf.cast(hp.encoder_depth, tf.float32))
    sc = tf.nn.softmax(e, axis=-1)

    # [B, N, A]
    r = tf.matmul(sc, v)

    return r, sc


def GLU(inputs, depth, k=7, activation=None, scope='GLU'):
  with tf.variable_scope(scope):
    conv = tf.layers.conv1d(
      inputs,
      filters=depth*2,
      kernel_size=k,
      activation=activation,
      padding='same',
      name='conv')

    x, gate = tf.split(conv, 2, axis=-1)
    if activation: x = activation(x)
    gate = tf.nn.sigmoid(gate)

    return x * gate


def gffw(x, depth, scope='gffw'):
  with tf.variable_scope(scope, reuse=REUSE):
    o = GLU(x, depth, k=hp.gffw_conv_k, activation=tf.nn.leaky_relu, scope='GLU')
    o = tf.layers.conv1d(o, depth, 1, padding='same', name='conv_pointwise')
  return o


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

  def cal_angle(position, hid_idx):
      return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
  
  def get_posi_angle_vec(position):
      return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
  
  sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
  
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
  
  # zero vector for padding dimension
  if padding_idx is not None: sinusoid_table[padding_idx] = 0.
  
  # [1, K, D]
  return tf.expand_dims(tf.convert_to_tensor(sinusoid_table, dtype=tf.float32), axis=0)


def get_attn_mask(xlen, max_xlen, ylen=None, max_ylen=None):
  if ylen is None and max_ylen is None: ylen, max_ylen = xlen, max_xlen
  x_unary = tf.expand_dims(tf.sequence_mask(xlen, max_xlen, dtype=tf.float32), 1)
  y_unary = tf.expand_dims(tf.sequence_mask(ylen, max_ylen, dtype=tf.float32), 1)
  mask = batch_dot(tf.transpose(x_unary, [0, 2, 1]), y_unary)
  return mask


def encoder_sa(x, x_len, f0, c0, y_len, is_training, scope='encoder'):
  depth = hp.encoder_depth

  with tf.variable_scope(scope, reuse=REUSE):
    # prenet
    if hp.txt_use_posenc:
      x = tf.layers.dense(x, depth, None, name='prenet')
      if hp.encoder_dropout:
        x = tf.layers.dropout(x, rate=0.2, training=is_training, name='dropout')

    '''multi-head self-attn: acoustics => global prosody'''
    slf_attns = []
    max_xlen = tf.shape(x)[-2]    # dynamic padded length N
    slf_mask = get_attn_mask(x_len, max_xlen)
    # D: 256 -> 64*4 -> 256
    for i in range(hp.encoder_attn_layers):
      # multi-head sa
      rs, attns = [], []
      #x = tf.keras.layers.LayerNormalization()(x)  # pre-norm
      for h in range(hp.encoder_attn_nhead):
        r, sc = dot_attn(x, x, slf_mask, depth // hp.encoder_attn_nhead, scope=f'sa_{i}_{h}')
        rs.append(r) ; attns.append(sc)
      slf_attns.append(attns)

      # combine multi-head
      sa = tf.layers.dense(tf.concat(rs, axis=-1), depth, name=f'proj_sa_{i}')
      if hp.encoder_dropout:
        sa = tf.layers.dropout(sa, rate=0.2, training=is_training, name='dropout')

      # transform (gffw)
      x = x + gffw(x + sa, depth, scope=f'gffw_sa_{i}')
      #x = tf.keras.layers.LayerNormalization()(x)    # post-norm

    ''' fusenet '''
    crx_attns = []
    f0_r = c0_r = f0_r_pred = c0_r_pred = 0.0
    if hp.encoder_fusenet:
      f0_r_pred = conv_stack(x, 2, hp.var_prednet_conv_k, hp.var_prednet_depth, hp.var_prednet_depth, activation=tf.nn.leaky_relu, scope='ca_f0_prednet')
      c0_r_pred = conv_stack(x, 2, hp.var_prednet_conv_k, hp.var_prednet_depth, hp.var_prednet_depth, activation=tf.nn.leaky_relu, scope='ca_c0_prednet')
      if is_training:
        max_ylen = tf.shape(f0)[-2]   # dynamic padded length T
        crx_mask = get_attn_mask(x_len, max_xlen, y_len, max_ylen)

        # [B, N, 256] cross attn [B, T, 64]
        f0_r, sc = dot_attn(x, f0, crx_mask, hp.var_prednet_depth, scope='ca_f0')
        crx_attns.append(sc)
        c0_r, sc = dot_attn(x, c0, crx_mask, hp.var_prednet_depth, scope='ca_c0')
        crx_attns.append(sc)
      
      # combine f0 & c0
      if is_training: f = tf.layers.dense(tf.concat([f0_r,      c0_r],      axis=-1), depth, name='proj_ca')
      else:           f = tf.layers.dense(tf.concat([f0_r_pred, c0_r_pred], axis=-1), depth, name='proj_ca')
      if hp.encoder_dropout:
        f = tf.layers.dropout(f, rate=0.2, training=is_training, name='dropout')

      # combine & transform (gffw)
      x = x + gffw(tf.concat([x, f], axis=-1), depth, scope=f'gffw_ca')

    return x, (slf_attns, crx_attns), ((f0_r, f0_r_pred), (c0_r, c0_r_pred))
