# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import encoder, decoder, converter
import numpy as np
import tensorflow as tf
from utils import *

class Graph:
    def __init__(self, training=True):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Data Feeding
            # x: Text. (N, T_x) 
            # y1: Reduced melspectrogram. (N, T_y//r, n_mels*r)
            # y2: Reduced dones. (N, T_y//r,)
            # z: Magnitude. (N, T_y, n_fft//2+1)
            if training:
                self.x, self.y1, self.y2, self.z, self.num_batch = get_batch()
            else: # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(None, hp.T_x))
                self.y1 = tf.placeholder(tf.float32, shape=(None, hp.T_y//hp.r, hp.n_mels*hp.r))

            # Get decoder inputs: feed last frames only (N, T_y//r, n_mels)
            self.decoder_inputs = tf.concat((tf.zeros_like(self.y1[:, :1, -hp.n_mels:]), self.y1[:, :-1, -hp.n_mels:]), 1)

            # Networks
            with tf.variable_scope("net"):
                # Encoder
                self.keys, self.vals = encoder(self.x,
                                               training=training,
                                               scope="encoder",
                                               reuse=None) # (N, T_x, E), (N, T_x, E)
                
                # Decoder 
                self.mels, self.dones, self.alignments = decoder(self.decoder_inputs,
                                                                  self.keys,
                                                                  self.vals,
                                                                  training=training,
                                                                  scope="decoder",
                                                                  reuse=None) # (N, T_y/r, n_mels*r), (N, T_y/r, 2), (N, T_y, T_x)
                # Restore shape
                self.mel_inputs = tf.reshape(self.mels, (hp.batch_size, hp.T_y, hp.n_mels))

                # Converter
                self.mags = converter(self.mel_inputs,
                                          training=training,
                                          scope="converter",
                                          reuse=None) # (N, T_y//r, (1+n_fft//2)*r)
            if training:
                # Loss
                self.loss1_mae = tf.reduce_mean(tf.abs(self.mels - self.y1))
                self.loss1_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dones, labels=self.y2)
                self.loss2 = tf.abs(self.mags - self.z)
                self.loss = self.loss1_mae + self.loss1_ce + self.loss2

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped =  [(tf.clip_by_norm(
                                    tf.clip_by_value(grad, -1.*hp.max_grad_val, hp.max_grad_val), var),
                                    hp.max_grad_norm
                                                 ) for grad, var in self.gvs]
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
                   
                # Summary
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('loss1_mae', self.loss1_mae)
                tf.summary.scalar('loss1_ce', self.loss1_ce)
                tf.summary.scalar('loss2', self.loss2)
                
                self.merged = tf.summary.merge_all()
if __name__ == '__main__':
    g = Graph(); print("Training Graph loaded")
    
    with g.graph.as_default():
        # # Load vocabulary
        # char2idx, idx2char = self.char2idx, self.idx2char
        
        # Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)
                
                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step) 
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

                # plot alignments
                gs, al = sess.run([g.global_step, g.alignments])
                plot_alignment(al[0].T, gs)

    print("Done")
