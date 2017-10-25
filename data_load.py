# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import csv
import os
import unicodedata

def text_normalize(sent):
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    normalized = re.sub("[^ a-z']", "", _strip_accents(sent).lower())

    return normalized

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts, mels, mags = [], [], []
    metadata = os.path.join(hp.data, 'metadata.csv')
    for line in codecs.open(metadata, 'r', 'utf-8'):
        fname, _, sent = line.strip().split("|")
        sent = text_normalize(sent) + "E" # text normalization, E: EOS
        if len(sent) <= hp.T_x:
            sent += "P"*(hp.T_x-len(sent))
            texts.append(np.array([char2idx[char] for char in sent], np.int32).tostring())
            mels.append(os.path.join(hp.data, "mels", fname + ".npy"))
            mags.append(os.path.join(hp.data, "mags", fname + ".npy"))

    return texts, mels, mags


def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _mels, _mags = load_train_data() # bytes
        
        # Calc total batch count
        num_batch = len(_texts) // hp.batch_size
         
        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts)
        mels = tf.convert_to_tensor(_mels)
        mags = tf.convert_to_tensor(_mags)
         
        # Create Queues
        text, mel, mag = tf.train.slice_input_producer([texts, mels, mags], shuffle=True)

        # Decoding to float32
        text = tf.decode_raw(text, tf.float32) # (T_x,)
        mel = tf.transpose(tf.decode_raw(tf.read_file(mel), tf.float32)) # (T_y/r, n_mels*r)
        mag = tf.transpose(tf.decode_raw(tf.read_file(mag), tf.float32)) # (T_y, 1+n_fft/2)

        # create batch queues
        xs, ys, zs = tf.train.batch([text, mel, mag],
                                shapes=[(hp.T_x,), (hp.T_y//hp.r, hp.n_mels*hp.r), (hp.T_y, 1+hp.n_fft//2)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=False)

    return xs, ys, zs, num_batch


