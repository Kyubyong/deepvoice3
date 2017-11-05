# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

import os

from scipy.io.wavfile import write

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import spectrogram2wav
from data_load import load_test_data
# import librosa.display
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from utils import *

def synthesize():
    # Load data
    X = load_test_data()

    # Load graph
    g = Graph(training=False); print("Graph loaded")

    # Inference
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")
             
            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]

            # Synthesize
            file_id = 1
            for i in range(0, len(X), hp.batch_size):
                x = X[i:i+hp.batch_size]

                # Get melspectrogram
                mel_output = np.zeros((hp.batch_size, hp.T_y//hp.r, hp.n_mels*hp.r), np.float32)
                decoder_output = np.zeros((hp.batch_size, hp.T_y//hp.r, hp.embed_size), np.float32)
                prev_max_attentions = np.zeros((hp.batch_size,), np.int32)
                max_attentions = np.zeros((hp.batch_size, hp.T_y//hp.r))
                alignments = np.zeros((hp.T_x, hp.T_y//hp.r), np.float32)
                for j in range(hp.T_y//hp.r):
                    _mel_output, _decoder_output, _max_attentions, _alignments = \
                        sess.run([g.mel_output, g.decoder_output, g.max_attentions, g.alignments],
                                  {g.x: x,
                                   g.y1: mel_output,
                                   g.prev_max_attentions: prev_max_attentions})
                    mel_output[:, j, :] = _mel_output[:, j, :]
                    decoder_output[:, j, :] = _decoder_output[:, j, :]
                    alignments[:, j] = _alignments[0].T[:, j]
                    prev_max_attentions = _max_attentions[:, j]
                    max_attentions[:, j] = _max_attentions[:, j]
                plot_alignment(alignments[::-1, :], "sanity-check", 0)

                # Get magnitude
                mags = sess.run(g.mag_output, {g.decoder_output: decoder_output})

                # Generate wav files
                if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
                for mag in mags:
                    print("file id=", file_id)
                    # generate wav files
                    mag = mag*hp.mag_std + hp.mag_mean # denormalize
                    audio = spectrogram2wav(np.power(10, mag) ** hp.sharpening_factor)
                    write(hp.sampledir + "/{}_{}.wav".format(mname, file_id), hp.sr, audio)
                    file_id += 1
                                          
if __name__ == '__main__':
    synthesize()
    print("Done")
    
    
