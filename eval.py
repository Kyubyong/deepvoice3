# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data

def eval():
    # Load data
    texts, mels, _, mags = load_data(training=False)

    x = np.array([np.fromstring(text, np.int32) for text in texts])
    y1 = np.array([np.load(mel) for mel in mels])
    z = np.array([np.load(mag) for mag in mags])

    # Padding
    x = np.array([np.pad(xx, ((0, hp.Tx),), "constant")[:hp.Tx] for xx in x])
    y1 = np.array([np.pad(xx, ((0, hp.Ty), (0, 0)), "constant")[:hp.Ty] for xx in y1])
    z = np.array([np.pad(xx, ((0, hp.Ty), (0, 0)), "constant")[:hp.Ty] for xx in z])

    # Load graph
    g = Graph(training=False); print("Graph loaded")

    # Inference
    with g.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()

            # Restore parameters
            saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

            # Writer
            writer = tf.summary.FileWriter(hp.logdir, sess.graph)

            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]

            # Get melspectrogram
            mel_output = np.zeros((hp.batch_size, hp.Ty // hp.r, hp.n_mels * hp.r), np.float32)
            decoder_output = np.zeros((hp.batch_size, hp.Ty // hp.r, hp.embed_size), np.float32)
            alignments_li = np.zeros((hp.dec_layers, hp.Tx, hp.Ty//hp.r), np.float32)
            prev_max_attentions_li = np.zeros((hp.dec_layers, hp.batch_size), np.int32)
            for j in range(hp.Ty // hp.r):
                _gs, _mel_output, _decoder_output, _max_attentions_li, _alignments_li = \
                    sess.run([g.global_step, g.mel_output, g.decoder_output, g.max_attentions_li, g.alignments_li],
                             {g.x: x,
                              g.y1: mel_output,
                              g.prev_max_attentions_li:prev_max_attentions_li})
                mel_output[:, j, :] = _mel_output[:, j, :]
                decoder_output[:, j, :] = _decoder_output[:, j, :]
                alignments_li[:, :, j] = np.array(_alignments_li)[:, :, j]
                prev_max_attentions_li = np.array(_max_attentions_li)[:, :, j]

            # Get magnitude
            mag_output = sess.run(g.mag_output, {g.decoder_output: decoder_output})

            # Loss
            mel_output = np.reshape(mel_output, (hp.batch_size, hp.Ty, hp.n_mels))
            eval_loss_mels = np.mean(np.abs(mel_output - y1))
            eval_loss_mags = np.mean(np.abs(mag_output - z))
            eval_loss = eval_loss_mels + eval_loss_mags

            # Generate the first wav file
            sent = "".join(g.idx2char[xx] for xx in x[-4]).split("E")[0]
            wav = spectrogram2wav(mag_output[-4])
            wav = np.expand_dims(wav, 0)

            # Summary
            tf.summary.scalar("Eval_Loss/mels", eval_loss_mels)
            tf.summary.scalar("Eval_Loss/mags", eval_loss_mags)
            tf.summary.scalar("Eval_Loss/LOSS", eval_loss)
            tf.summary.text("Sent", tf.convert_to_tensor(sent))
            tf.summary.audio("audio sample", wav, hp.sr, max_outputs=1)
            tf.summary.histogram("mag/output", mag_output)
            tf.summary.histogram("mel/output", mel_output)
            tf.summary.histogram("mag/target", z)
            tf.summary.histogram("mel/target", y1)
            tf.summary.image("alignments", np.expand_dims(alignments_li, -1), max_outputs=len(alignments_li))

            merged = tf.summary.merge_all()
            writer.add_summary(sess.run(merged), global_step=_gs)
            writer.close()

if __name__ == '__main__':
    eval()
    print("Done")


