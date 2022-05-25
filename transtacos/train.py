import os
from time import time, sleep
from pprint import pformat
from argparse import ArgumentParser
import traceback

import tensorflow as tf
import numpy as np

import hparam as hp
from models.tacotron import Tacotron
from data import DataFeeder
from text.text import sequence_to_phoneme
from audio import save_wav, inv_spec
from utils import *


for gpu in tf.config.experimental.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_XLA_FLAGS']         = '--tf_xla_cpu_global_jit'
os.environ['XLA_FLAGS']            = '--xla_hlo_profile'

np.random.seed           (hp.randseed)
tf.random.set_random_seed(hp.randseed)


def train(args):
  # Logg Folder
  log_dir = os.path.join(args.base_dir, args.name)
  os.makedirs(log_dir, exist_ok=True)
  log_init(os.path.join(log_dir, 'train.log'))

  ckpt_path = os.path.join(log_dir, 'model.ckpt')
  log('Checkpoint path: %s' % ckpt_path)
  input_path = os.path.join(args.base_dir, args.input)
  log('Loading training data from: %s' % input_path)
  log('Hyperparams:')
  log(pformat({k: getattr(hp, k) for k in dir(hp) if not k.startswith('__')}, indent=2))

  # DataFeeder
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder'):
    feeder = DataFeeder(coord, input_path, hp)

  # Model
  with tf.variable_scope('model'):
    model = Tacotron(hp)
    model.initialize(feeder.text_lengths,
                     feeder.text, feeder.prds,
                     feeder.spec_lengths,
                     feeder.mel_targets, feeder.mag_targets, feeder.f0_targets, feeder.c0_targets,
                     feeder.stop_token_targets)
    model.add_loss()
    model.add_optimizer()
    model.add_stats()
  param_count = sum([np.prod(v.get_shape()) for v in tf.trainable_variables()])
  log(f'param_cnt = {param_count}')
  
  # Bookkeeping
  step = 0      # local step
  time_window = ValueWindow(100)      # for perfcount
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=hp.max_ckpt)

  # Train!
  with tf.Session() as sess:
    try:
      sw = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      # Restore from a checkpoint if available
      ckpt_state = tf.train.get_checkpoint_state(log_dir)
      if ckpt_state is not None:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        log('Resuming from checkpoint: %s' % ckpt_state.model_checkpoint_path)
      else:
        log('Starting new training run')

      feeder.start_in_session(sess)
      while not coord.should_stop():
        t = time()
        step, loss, opt = sess.run([model.global_step, model.loss, model.optimize])
        time_window.append(time() - t)
        loss_window.append(loss)
        log('Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (step, time_window.average, loss, loss_window.average))

        if loss > 300 or np.isnan(loss):
          log('Loss exploded to %.05f at step %d!' % (loss, step))
          raise Exception('Loss Exploded')

        if step % args.summary_interval == 0:
          log('Writing summary at step: %d' % step)
          sw.add_summary(sess.run(model.stats), step)

        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (ckpt_path, step))
          saver.save(sess, ckpt_path, global_step=step)
          log('Saving audio and alignment...')

          if hp.g2p == 'seq':
            (text, mel, mag, alignment, spec_len, mel_r, mag_r, mel_loss, mag_loss) = sess.run([
              model.text[0], 
              model.mel_outputs[0], model.mag_outputs[0], model.alignments[0], model.spec_lengths[0],
              model.mel_targets[0], model.mag_targets[0], model.mel_loss, model.mag_loss])
            log('Input:')
            log(f'  seq: {text}')
            log(f'  phs: {sequence_to_phoneme(text)}')
          elif hp.g2p == 'syl4':
            (text, prds_o, prds_r, mel, mag, alignment, spec_len, mel_r, mag_r, mel_loss, mag_loss) = sess.run([
              model.text[0], model.prds_out[0], model.prds[0], 
              model.mel_outputs[0], model.mag_outputs[0], model.alignments[0], model.spec_lengths[0],
              model.mel_targets[0], model.mag_targets[0], model.mel_loss, model.mag_loss])
            CVVx, T = text.T.tolist()
            log('Input:')
            log(f'  text: {sequence_to_phoneme(CVVx)}')
            log(f'  tone: {"".join([str(t) for t in T])}')
            log(f'  prds: {"".join([str(p) for p in prds_r])}')
            log(f'  pred: {"".join([str(p) for p in prds_o])}')

          mel, mag, mel_r, mag_r = [m[:spec_len,:].T for m in [mel, mag, mel_r, mag_r]]
          save_wav(inv_spec(mag), os.path.join(log_dir, 'step-%d-audio.wav' % step))
          plot_specs([mel, mag, mel_r, mag_r], os.path.join(log_dir, 'step-%d-specs.png' % step),
                      info=f'{time_string()}, mel_loss={mel_loss:.5f}, mag_loss={mag_loss:.5f}')
          plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                         info='%s, step=%d, loss=%.5f' % (time_string(), step, loss))

        if step >= hp.max_steps + 10:
          print('[Train] Done')
          sleep(5)
          break

    except Exception as e:
      log('Exiting due to exception: %s' % e)
      traceback.print_exc()
      coord.request_stop(e)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('.'))
  parser.add_argument('--input',    default='preprocessed/train.txt')
  parser.add_argument('--name',     default='transtacos', help='Name of the run, used for logging.')
  parser.add_argument('--summary_interval',    type=int, default=1000, help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1500, help='Steps between writing checkpoints.')
  args = parser.parse_args()

  train(args)
