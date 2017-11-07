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
                print("".join(g.idx2char[xx] for xx in x[0]))

                # Decoder
                _mel_output = np.zeros((hp.batch_size, hp.Ty, hp.n_mels), np.float32)
                _decoder_output = np.zeros((hp.batch_size, hp.Ty, hp.embed_size), np.float32)
                _prev_max_attentions = np.zeros((hp.batch_size,), np.int32)
                _max_attentions = np.zeros((hp.batch_size, hp.Ty))
                _alignments = np.zeros((hp.dec_layers, hp.Tx, hp.Ty), np.float32)
                for j in range(hp.Ty):
                    mel_output_, decoder_output, max_attentions, alignments = \
                        sess.run([g.mel_output_, g.decoder_output, g.max_attentions, g.alignments],
                                  {g.x: x,
                                   g.y1: _mel_output,
                                   g.prev_max_attentions: _prev_max_attentions})
                    # mel_output_ (N, Ty, n_mels*r)
                    if j % hp.r == 0:
                        for k, ary in enumerate(np.split(mel_output_[:, j, :], hp.r, -1)):
                            _mel_output[:, j+k, :] = ary
                    _decoder_output[:, j, :] = _decoder_output[:, j, :]
                    _alignments[:, :, j] = alignments[:, :, j]
                    _prev_max_attentions = _max_attentions[:, j]
                    _max_attentions[:, j] = max_attentions[:, j]

                # sanity-check
                plot_alignment(_alignments, "sanity-check", "")

                # Converter
                print(np.mean(decoder_output))
                print(np.max(decoder_output))
                print(np.min(decoder_output))
                mags = sess.run(g.mags, {g.decoder_output: decoder_output})

                # Generate wav files
                if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
                for mag in mags:
                    print("file id=", file_id)
                    # generate wav files
                    print(np.max(mag), np.min(mags))
                    mag = mag*hp.mag_std + hp.mag_mean # denormalize
                    print(np.max(mag), np.min(mag))
                    print(np.max(np.power(10, mag)), np.min(np.power(10, mag)))
                    audio = spectrogram2wav(np.power(10, mag) ** hp.sharpening_factor)
                    write(hp.sampledir + "/{}_{}.wav".format(mname, file_id), hp.sr, audio)
                    file_id += 1
                                          
if __name__ == '__main__':
    synthesize()
    print("Done")
    
    
