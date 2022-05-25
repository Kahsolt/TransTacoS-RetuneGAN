import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, ResidualWrapper

from .modules import *
from .custom_decoder import *
from .rnn_wrappers import *
from .attention import *
from text.symbols import get_vocab_size
from audio import inv_spec
from utils import log


class Tacotron():
  
  def __init__(self, hparams):
    self._hparams = hparams
    self.global_step = tf.Variable(0, name='global_step', trainable=False)


  def initialize(self, text_lengths, text, prds=None,
                 spec_lengths=None, mel_targets=None, mag_targets=None, f0_targets=None, c0_targets=None,
                 stop_token_targets=None):
    with tf.variable_scope('inference'):
      hp = self._hparams
      is_training = mel_targets is not None
      B = batch_size = tf.shape(text)[0]

      print('text_lengths.shape:',  text_lengths.shape)
      print('text.shape:',          text.shape)
      if is_training:
        print('prds.shape:',               prds.shape)
        print('spec_lengths.shape:',       spec_lengths.shape)
        print('mel_targets.shape:',        mel_targets.shape)
        print('mag_targets.shape:',        mag_targets.shape)
        print('f0_targets.shape:',         f0_targets.shape)
        print('c0_targets.shape:',         c0_targets.shape)
        print('stop_token_targets.shape:', stop_token_targets.shape)
      log(f'[Tacotron] vocab size {get_vocab_size()}')

      # Embeddings
      # 不用零pad好像关系也不大
      zero_embedding_pad      = tf.constant(0, shape=[1, hp.embed_depth],      dtype=tf.float32, name='zero_embedding_pad')
      zero_embedding_pad_half = tf.constant(0, shape=[1, hp.encoder_depth//2], dtype=tf.float32, name='zero_embedding_pad_half')

      '''位置编码嵌入'''
      PE_table = get_sinusoid_encoding_table(max(hp.maxlen_text, hp.maxlen_spec), hp.posenc_depth)   # 姑且统一深度，使用concat方式

      '''语言学特征嵌入'''
      if hp.g2p == 'seq':
        E_text = tf.get_variable('E_text', [get_vocab_size(), hp.embed_depth], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
        
        # seq
        text_embd = tf.nn.embedding_lookup(E_text, text)
        embd_out = text_embd 
      
      elif hp.g2p == 'syl4':
        E_text = tf.get_variable('E_text', [get_vocab_size(), hp.embed_depth], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
        E_tone = tf.get_variable('E_tone', [hp.n_tone,        hp.embed_depth], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
        E_prds = tf.get_variable('E_prds', [hp.n_prds,        hp.embed_depth], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
        
        # syl4
        CVVx, T = [tf.squeeze(p, axis=-1) for p in tf.split(text, 2, axis=-1)]    # [B, T, 2] => 2 * [B, T] 
        phone_embd = tf.nn.embedding_lookup(E_text, CVVx)
        tone_embd  = tf.nn.embedding_lookup(E_tone, T)
        text_embd = phone_embd + tone_embd

        # prds
        prds_prob = conv_stack(text_embd, 3, hp.prdsnet_conv_k, hp.prdsnet_depth, hp.n_prds, activation=tf.nn.relu, scope='prdsnet')
        prds_out  = tf.argmax(prds_prob, axis=-1)
        if is_training: prds_embd = tf.nn.embedding_lookup(E_prds, prds)
        else:           prds_embd = tf.nn.embedding_lookup(E_prds, prds_out)

        embd_out = text_embd + prds_embd
      
      if hp.embed_dropout:
        embd_out = tf.layers.dropout(embd_out,  rate=0.2, training=is_training, name='dropout_N')
      if is_training:
        embd_out = gaussian_noise(embd_out, is_training)

      if hp.encoder_type == 'sa':
        if hp.txt_use_posenc:
          N_pos_embd_out = tf.tile(PE_table[:, :tf.shape(embd_out)[1], :], (B, 1, 1))
          embd_out = tf.concat([embd_out, N_pos_embd_out], axis=-1)

        if is_training:
          '''声学特征离散嵌入'''
          E_f0 = tf.get_variable('E_f0', [hp.n_f0_bins, hp.var_embed_depth], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
          E_c0 = tf.get_variable('E_c0', [hp.n_c0_bins, hp.var_embed_depth], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))

          f0_embd = tf.nn.embedding_lookup(E_f0, f0_targets)
          c0_embd = tf.nn.embedding_lookup(E_c0, c0_targets)
          
          if hp.embed_dropout:
            f0_embd = tf.layers.dropout(f0_embd, rate=0.2, training=is_training, name='dropout_T')
            c0_embd = tf.layers.dropout(c0_embd, rate=0.2, training=is_training, name='dropout_T')
          if is_training:
            f0_embd = gaussian_noise(f0_embd, is_training)
            c0_embd = gaussian_noise(c0_embd, is_training)
          
          if hp.var_use_posenc:
            T_pos_embd_out = tf.tile(PE_table[:, :tf.shape(f0_targets)[-1], :], (B, 1, 1))
            f0_embd = tf.concat([f0_embd, T_pos_embd_out], axis=-1)
            c0_embd = tf.concat([c0_embd, T_pos_embd_out], axis=-1)
        else:
          f0_embd = c0_embd = None

      # Encoder
      if hp.encoder_type == 'sa':
        encoder_out, (slf_attn, crx_attn), ((f0_r, f0_r_pred), (c0_r, c0_r_pred)) = encoder_sa(embd_out, text_lengths, f0_embd, c0_embd, spec_lengths, is_training)
      elif hp.encoder_type == 'cb':
        encoder_out = cbhg(embd_out, text_lengths, hp.encoder_conv_K, [hp.encoder_depth//2, hp.encoder_depth], hp.encoder_depth, is_training)
      else: raise
      if is_training: encoder_out = gaussian_noise(encoder_out, is_training)

      # Decoder (layers specified bottom to top):                                # 将生成序列长度的控制问题转换为RNN迭代次数预测，参见`stop_projection`
      multi_rnn_cell = MultiRNNCell([                                            # 将inner_repr喂给RNN产出rnn_output
          ResidualWrapper(GRUCell(hp.decoder_depth))
            for  _ in range(hp.decoder_layers)
        ], state_is_tuple=True)                                                  # [N, T_in, decoder_depth=256]
      attention_mechanism = LocationSensitiveAttention(hp.attention_depth, encoder_out, text_lengths)  # [N, T_in, attn_depth=128]
      frame_projection = FrameProjection(hp.n_mel * hp.outputs_per_step)         # [N, T_out/r, M*r], 将concat([rnn_output,attn_context])投影为长度 r*n_mel 的向量、之后会reshape成r帧
      stop_projection = StopProjection(is_training, shape=hp.outputs_per_step)   # [N, T_out/r, r], 投影为r个标量、只要其中有一个大于0.5就认结束生成
      decoder_cell = TacotronDecoderWrapper(multi_rnn_cell, attention_mechanism, frame_projection, stop_projection, is_training)
      if is_training: helper = TacoTrainingHelper(batch_size, mel_targets, hp.n_mel, hp.outputs_per_step, self.global_step)
      else:           helper = TacoTestHelper(batch_size, hp.n_mel, hp.outputs_per_step)
      decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      (decoder_out, stop_token_out, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
         CustomDecoder(decoder_cell, helper, decoder_init_state),               # [N, T_out/r, M*r] 
         impute_finished=True, maximum_iterations=hp.max_iters)

      # Reshape outputs to be one output per entry
      mel_out = tf.reshape(decoder_out, [batch_size, -1, hp.n_mel])             # [N, T_out, M], mel用于参与loss计算、并不用于产生最终wav
      stop_token_out = tf.reshape(stop_token_out, [batch_size, -1])             # [N, T_out], 这个结果没有用、只是在decode时作参考而已
      alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

      # k = 7 should cover 1.5 mel-groups (reduce_factor)
      if hp.decoder_sew_layer:
        mel_out += tf.layers.conv1d(mel_out, hp.n_mel, 7, padding='same', name='sew_up_layer')

      # Posnet
      x = mel_out[:,:,:hp.n_mel_low]
      x = tf.layers.dense(x, hp.posnet_depth//4, name='posnet1')
      x = tf.nn.leaky_relu(x)
      x = tf.layers.dense(x, hp.posnet_depth//2, name='posnet2')
      x = tf.nn.leaky_relu(x)
      x = tf.layers.dense(x, hp.posnet_depth,    name='posnet3')
      x = tf.nn.leaky_relu(x)
      mag_out = tf.concat([tf.layers.dense(s, (hp.n_freq-1)//hp.posnet_ngroup, name=f'posnet4_{i}') 
                           for i, s in enumerate(tf.split(x, hp.posnet_ngroup, axis=-1))], axis=-1)

      # data in
      self.text_lengths       = text_lengths
      self.text               = text
      self.prds               = prds
      self.spec_lengths       = spec_lengths
      self.mel_targets        = mel_targets
      self.mag_targets        = mag_targets
      self.stop_token_targets = stop_token_targets
      # data out
      self.prds_prob          = prds_prob
      self.prds_out           = prds_out
      self.mel_outputs        = mel_out
      self.mag_outputs        = mag_out
      self.stop_token_outputs = stop_token_out
      self.alignments         = alignments
      # misc
      if is_training:
        # NOTE: must get `._ratio` after  `TacoTrainingHelper.initialize()`
        self.tfr = helper._ratio
      if hp.encoder_type == 'sa':
        self.slf_attn = slf_attn
        self.crx_attn = crx_attn
        self.f0_r = f0_r
        self.f0_r_pred = f0_r_pred
        self.c0_r = c0_r
        self.c0_r_pred = c0_r_pred
    
      def get_cosine_sim(x):
        dot = tf.matmul(x, x, transpose_b=True)
        n = tf.norm(x, axis=-1, keepdims=True)
        norm = tf.matmul(n, n, transpose_b=True)
        sim = dot / (norm + 1e-8)
        return sim

      self.E_text = E_text
      self.E_text_sim = get_cosine_sim(E_text)
      if hp.g2p == 'syl4':
        self.E_tone = E_tone
        self.E_tone_sim = get_cosine_sim(E_tone)
        self.E_prds = E_prds
        self.E_prds_sim = get_cosine_sim(E_prds)

      log('Initialized TrasTacoS Model: ')
      log(f'  embd out:                {embd_out.shape}')
      if hp.g2p == 'syl4':
        log(f'     syl4 embd:            {text_embd.shape}')
        log(f'     tone embd:            {tone_embd.shape}')
        log(f'     prds embd:            {prds_embd.shape}')
      if hp.encoder_type == 'sa' and is_training:
        log(f'   f0 embd:               {f0_embd.shape}')
        log(f'   c0 embd:               {c0_embd.shape}')
      log(f'  encoder out:             {encoder_out.shape}')
      log(f'  decoder out (r frames):  {decoder_out.shape}')
      log(f'  mel out (1 frame):       {mel_out.shape}')
      log(f'  stoptoken out:           {stop_token_out.shape}')
      log(f'  mag out:                 {mag_out.shape}')
      log(f'  E_text:                  {E_text.shape}')
      if hp.g2p == 'syl4':
        log(f'  E_tone:                  {E_tone.shape}')
        log(f'  E_prds:                  {E_prds.shape}')


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize() must have been called.'''

    hp = self._hparams
    with tf.variable_scope('loss'):
      self.mel_loss = tf.reduce_mean(tf.abs(self.mag_targets - self.mag_outputs))
      self.mag_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
      if hp.encoder_type == 'sa' and hp.encoder_fusenet:
        self.f0_loss  = tf.reduce_mean(tf.square(self.f0_r - self.f0_r_pred))
        self.c0_loss  = tf.reduce_mean(tf.square(self.c0_r - self.c0_r_pred))
      else:
        self.f0_loss  = 0.0
        self.c0_loss  = 0.0
      if hp.g2p == 'syl4':
        self.prds_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.prds, logits=self.prds_prob))
      else:
        self.prds_loss = 0.0
      if hp.g2p == 'seq':
        self.sim_loss      = tf.reduce_mean(tf.abs((1.0 - tf.eye(get_vocab_size())) * self.E_text_sim)) * hp.sim_weight
      else:
        self.sim_loss      = tf.add_n([tf.reduce_mean(tf.abs((1.0 - tf.eye(get_vocab_size())) * self.E_text_sim)),
                                       tf.reduce_mean(tf.abs((1.0 - tf.eye(hp.n_prds))        * self.E_prds_sim))]) * hp.sim_weight
      self.stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.stop_token_targets, logits=self.stop_token_outputs))
      self.reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * hp.reg_weight

      self.loss = (self.prds_loss +
                   self.mel_loss + 
                   self.mag_loss + 
                   self.f0_loss +
                   self.c0_loss +
                   self.sim_loss +
                   self.stop_token_loss +
                   self.reg_loss)

  def add_optimizer(self):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss() must have been called.'''

    with tf.variable_scope('optimizer'):
      hp = self._hparams

      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, self.global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2, hp.adam_eps)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = tuple([g for g in gradients if g is not None])   # FIXME: do not know why
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)    # 防止梯度爆炸

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=self.global_step)


  def add_stats(self):
    with tf.variable_scope('stats'):
      hp = self._hparams

      tf.summary.histogram('mel_outputs', self.mel_outputs)
      tf.summary.histogram('mel_targets', self.mel_targets)
      tf.summary.histogram('mag_outputs', self.mag_outputs)
      tf.summary.histogram('mag_targets', self.mag_targets)

      tf.summary.scalar('learning_rate', self.learning_rate)
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('tfr', self.tfr)
      tf.summary.scalar('mel_loss', self.mel_loss)
      tf.summary.scalar('mag_loss', self.mag_loss)
      if hp.g2p == 'syl4':
        tf.summary.scalar('prds_loss', self.prds_loss)
      if hp.encoder_type == 'sa':
        tf.summary.scalar('f0_loss', self.f0_loss)
        tf.summary.scalar('c0_loss', self.c0_loss)
      tf.summary.scalar('sim_loss',        self.sim_loss)
      tf.summary.scalar('stop_token_loss', self.stop_token_loss)
      tf.summary.scalar('reg_loss', self.reg_loss)
      
      gradient_norms = [tf.norm(grad) for grad in self.gradients]
      tf.summary.histogram('gradient_norm', gradient_norms)
      tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))

      raw = tf.numpy_function(inv_spec, [tf.transpose(self.mag_targets[0])], tf.float32)
      gen = tf.numpy_function(inv_spec, [tf.transpose(self.mag_outputs[0])], tf.float32)
      tf.summary.audio('raw', tf.expand_dims(raw, 0), hp.sample_rate, 1)
      tf.summary.audio('gen', tf.expand_dims(gen, 0), hp.sample_rate, 1)

      expand_dims = lambda x: tf.expand_dims(tf.expand_dims(x, 0), -1)
      tf.summary.image('alignments', expand_dims(self.alignments[0]))
      tf.summary.image('E_text_sim', expand_dims(self.E_text_sim))
      if hp.g2p != 'seq':
        tf.summary.image('E_tone_sim', expand_dims(self.E_tone_sim))
        tf.summary.image('E_prds_sim', expand_dims(self.E_prds_sim))
      if hp.encoder_type == 'sa':
        for i in range(hp.encoder_attn_layers):
          for j in range(hp.encoder_attn_nhead):
            tf.summary.image(f'slf_attn_{i}{j}', expand_dims(self.slf_attn[i][j][0]))
        if self.crx_attn:
          for i in range(2):
            tf.summary.image(f'crx_attn_{i}', expand_dims(self.crx_attn[i][0]))
      
      #tf.summary.image('mel_out', expand_dims(self.mel_outputs[0]))

      self.stats = tf.summary.merge_all()


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
