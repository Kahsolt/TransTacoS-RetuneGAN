from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper, Decoder
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

import hparam as hp


# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacoTestHelper(Helper):
  def __init__(self, batch_size, output_dim, r):
    with tf.name_scope('TacoTestHelper'):
      self._batch_size = batch_size
      self._output_dim = output_dim     # output_dim==n_mels
      self._reduction_factor = r

  @property
  def batch_size(self):         # IGNORED
    return self._batch_size

  @property
  def token_output_size(self):  # IGNORED
    return self._reduction_factor     # 每次吐r帧

  @property
  def sample_ids_shape(self):   # IGNORED
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):   # IGNORED
    return np.int32

  def sample(self, time, outputs, state, name=None):    # IGNORED
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them
  
  def initialize(self, name=None):    # init 1d vetor of False
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))    # append <GO>

  def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name=None):
    '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
    with tf.name_scope('TacoTestHelper'):
      # A sequence is finished when the stop token probability is > 0.5
      # With enough training steps, the model should be able to predict when to stop correctly
      # and the use of stop_at_any = True would be recommended. If however the model didn't
      # learn to stop correctly yet, (stops too soon) one could choose to use the safer option
      # to get a correct synthesis
      # 难以通过判断产生了一个静音的mel帧来判定生成结束，所以用了一个与mel帧等长的stop_token向量
      # 一旦向量中(理论上应该靠近末端)出现了接近1.0的值即可认为生成结束，另参见`datafeeder._pad_stop_token_target()`
      # NOTE: 为了容错，不能只看stop_token_preds[-1]
      finished = tf.reduce_any(tf.cast(tf.round(stop_token_preds), tf.bool))

      # Feed last output frame as next input. outputs is [N, output_dim * r]
      next_inputs = outputs[:, -self._output_dim:]               # take last frame of a frame group
      return (finished, next_inputs, state)


class TacoTrainingHelper(Helper):
  def __init__(self, batch_size, targets, output_dim, r, global_step):
    # inputs is [N, T_in], targets is [N, T_out, D]
    with tf.name_scope('TacoTrainingHelper'):
      self._batch_size = batch_size
      self._output_dim = output_dim                 # =n_mels
      self._reduction_factor = r
      self._ratio = None
      self.global_step = global_step

      # Feed every r-th target frame as input
      self._targets = targets[:, r-1::r, :]         # 每个帧组的最后一帧, every r-th frame

      # Use full length for every target because we don't want to mask the padding frames
      num_steps = tf.shape(self._targets)[1]        # =max_timesetps         # FIXME: why cannot stop early?
      self._lengths = tf.tile([num_steps], [self._batch_size])

  @property
  def batch_size(self):          # IGNORED
    return self._batch_size

  @property
  def token_output_size(self):   # IGNORED
    return self._reduction_factor

  @property
  def sample_ids_shape(self):    # IGNORED
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):    # IGNORED
    return np.int32

  def sample(self, time, outputs, state, name=None):   # IGNORED
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def initialize(self, name=None):
    self._ratio = _teacher_forcing_ratio_decay(hp.tf_init, self.global_step)
    return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))    # append <GO>

  def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name='TacoTrainingHelper'):
    with tf.name_scope(name):
      finished = (time + 1 >= self._lengths)           # 训练时读到最后一帧mel就算结束，即这个batch的最大帧组长度

      if hp.tf_method == 'force':
        next_inputs = self._targets[:, time, :]
      elif hp.tf_method == 'random':
        next_inputs = tf.cond(tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
                lambda: self._targets[:, time, :],     
                lambda: outputs[:, -self._output_dim:])
      elif hp.tf_method == 'mix':
        next_inputs = self._ratio * self._targets[:, time, :] + (1 - self._ratio) * outputs[:, -self._output_dim:]
      else: raise ValueError

      return (finished, next_inputs, state)


def _go_frames(batch_size, output_dim):
  '''Returns all-zero <GO> frames for a given batch size and output dimension'''
  return tf.tile([[0.0]], [batch_size, output_dim])


def _teacher_forcing_ratio_decay(init_tfr, global_step):
  #################################################################
  # Narrow Cosine Decay:

  # Phase 1: tfr = 1
  # We only start learning rate decay after 10k steps

  # Phase 2: tfr in [0, 1]
  # decay reach minimal value at step ~280k

  # Phase 3: tfr = 0
  # clip by minimal teacher forcing ratio value (step >~ 280k)
  #################################################################
  # Compute natural cosine decay
  tfr = tf.train.cosine_decay(init_tfr,
          global_step=global_step - hp.tf_start_decay,  # tfr = 1 at step 10k, (original: 20000)
          decay_steps=hp.tf_decay,                      # tfr = 0 at step ~280k, (original: 200000)
          alpha=0.,                                     # tfr = 0% of init_tfr as final value
          name='tfr_cosine_decay')

  # force teacher forcing ratio to take initial value when global step < start decay step.
  # NOTE: narrow_tfr = global_step < 10000 ? init_tfr : tfr
  narrow_tfr = tf.cond(
          tf.less(global_step, tf.convert_to_tensor(hp.tf_start_decay)),   # original: 20000
          lambda: tf.convert_to_tensor(init_tfr),
          lambda: tfr)

  return narrow_tfr


class CustomDecoderOutput(
    namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
  pass


class CustomDecoder(Decoder):
  """Custom sampling decoder.

  Allows for stop token prediction at inference time
  and returns equivalent loss in training time.

  Note:
  Only use this decoder with Tacotron 2 as it only accepts tacotron custom helpers
  """

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize CustomDecoder.
    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.
    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    rnn_cell_impl.assert_like_rnncell(type(cell), cell)
    if not isinstance(helper, Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError("output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return CustomDecoderOutput(
        rnn_output=self._rnn_output_size(),
        token_output=self._helper.token_output_size,
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and the sample_ids_dtype from the helper.
    dtype = nest.flatten(self._initial_state)[0].dtype
    return CustomDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        tf.float32,
        self._helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.
    Args:
      name: Name scope for any created operations.
    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a custom decoding step.
    Enables for dyanmic <stop_token> prediction
    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.
    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "CustomDecoderStep", (time, inputs, state)):
      #Call outputprojection wrapper cell
      (cell_outputs, stop_token), cell_state = self._cell(inputs, state)

      #apply output_layer (if existant)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)

      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids,
          stop_token_preds=stop_token)

    outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)
    return (outputs, next_state, next_inputs, finished)
