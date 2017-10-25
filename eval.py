# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import codecs
import copy
import os

import librosa
from scipy.io.wavfile import write

from hyperparams import Hyperparams as hp
import numpy as np
from data_load import load_vocab, load_eval_data
import tensorflow as tf
from train import Graph
from utils import spectrogram2wav, restore_shape


def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X = load_eval_data() # texts
    print(X.shape)

    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
             
            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            print(hp.num_samples, hp.max_len//hp.r, hp.n_mels*hp.r)
            outputs1 = np.zeros((hp.num_samples, int(hp.max_len//hp.r), hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
            for j in range(int(hp.max_len//hp.r)):
                _outputs1 = sess.run(g.outputs1, {g.x: X, g.y: outputs1})
                outputs1[:, j, :] = _outputs1[:, j, :]
            outputs2 = sess.run(g.outputs2, {g.outputs1: outputs1})

    # Generate wav files
    if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
    with codecs.open(hp.sampledir + '/text.txt', 'w', 'utf-8') as fout:
        for i, (x, mag) in enumerate(zip(X, outputs2)):
            # write text
            fout.write(str(i) + "\t" + "".join(g.idx2char[idx] for idx in np.fromstring(x, np.int32) if idx != 0) + "\n")

            # restore shape
            mag = restore_shape(mag)

            # generate wav files
            mag = mag*hp.mag_std + hp.mag_mean # denormalize
            mag = np.exp(mag) ** hp.power # exponentiate

            print(np.mean(mag), "|", np.max(mag), np.min(mag))

            audio = spectrogram2wav(mag)
            write(hp.sampledir + "/{}_{}.wav".format(mname, i), hp.sr, audio)
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
