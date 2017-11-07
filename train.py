# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import encoder, decoder, converter
import tensorflow as tf
from utils import *
import time


class Graph:
    def __init__(self, training=True):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Data Feeding
            ## x: Text. (N, Tx), int32
            ## y1: log melspectrogram. (N, Ty, n_mels) float32
            ## y2: dones. (N, Ty,) int32
            ## z: log linear magnitude. (N, Ty, n_fft//2+1) float32
            if training:
                self.x, self.y1, self.y2, self.z, self.num_batch = get_batch()
                self.prev_max_attentions = tf.constant([0] * hp.batch_size)
            else:  # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.Tx))
                self.y1 = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.Ty, hp.n_mels))
                self.prev_max_attentions = tf.placeholder(tf.int32, shape=(hp.batch_size,))

            # Get decoder inputs: feed last frames only (N, Ty, n_mels)
            self.decoder_input = tf.concat((tf.zeros_like(self.y1[:, :1, :]), self.y1[:, :-1, :]), 1)

            # Networks
            with tf.variable_scope("net"):
                # Encoder. keys: (N, Tx, e), vals: (N, Tx, e)
                self.keys, self.vals = encoder(self.x,
                                               training=training,
                                               scope="encoder")

                print("# Decoder.")# mel_output_: # (N, Ty, n_mels*r), done_output: (N, Ty, 2),

                # decoder_output: (N, Ty, e), alignments: dec_layers * (Ty, Tx), keys_li :# dec_layers*(Tx, a)
                self.mel_output_, self.done_output, self.decoder_output, self.alignments, self.max_attentions = \
                    decoder(self.decoder_input,
                            self.keys,
                            self.vals,
                            self.prev_max_attentions,
                            training=training,
                            scope="decoder")

                # Reshape
                self.mel_output = self.mel_output_[:, ::hp.r, :]  # (N, Ty/r, n_mels*r)
                self.mel_output = tf.reshape(self.mel_output, (hp.batch_size, hp.Ty, hp.n_mels))  # (N, Ty, n_mels)

                # Activation. converter_input: (N, Ty, e)
                self.converter_input = glu(self.mel_output)

                # Converter. mag_output: (N, Ty, 1+n_fft//2)
                self.mag_output = converter(self.converter_input,
                                            training=training,
                                            scope="converter")
            if training:
                print("# Loss")
                self.loss1_mae = tf.reduce_mean(tf.abs(self.mel_output - self.y1))
                self.loss1_ce = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.done_output, labels=self.y2))
                self.loss2 = tf.reduce_mean(tf.abs(self.mag_output - self.z))
                self.loss = self.loss1_mae + self.loss1_ce + self.loss2

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                ## gradient clipping
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.gvs:
                    grad = tf.clip_by_value(grad, -1. * hp.max_grad_val, hp.max_grad_val)
                    grad = tf.clip_by_norm(grad, hp.max_grad_norm)
                    self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

                # Summary
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('loss1_mae', self.loss1_mae)
                tf.summary.scalar('loss1_ce', self.loss1_ce)
                tf.summary.scalar('loss2', self.loss2)

                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    start_time = time.time()
    g = Graph(); print("Training Graph loaded")
    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0)
        with sv.managed_session() as sess:
            # plot initial alignments
            al = sess.run(g.alignments)
            plot_alignment(al, 0, 0)  # (Tx, Ty/r)

            while 1:
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    gs, _ = sess.run([g.global_step, g.train_op])

                    # if gs > 0 and gs % 100 == 0:
                    #     # plot alignments
                    #     temp = sess.run(g.temp)
                    #     print(temp)

                    # Write checkpoint files at every epoch
                    if gs > 0 and gs % 1000 == 0:
                        sv.saver.save(sess, hp.logdir + '/model_gs_%dk' % (gs//1000))

                        # plot alignments
                        alignments = sess.run(g.alignments)
                        elapsed_time = time.time() - start_time
                        plot_alignment(alignments, str(gs//1000) + "k", elapsed_time)  # (Tx, Ty)

                # break
                if gs > hp.num_iterations: break
    print("Done")
