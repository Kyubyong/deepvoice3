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
import os
import unicodedata

def text_normalize(sent):
    '''Minimum text preprocessing'''
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    normalized = re.sub("[^ a-z']", "", _strip_accents(sent).lower())

    return normalized

def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts, mels, dones, mags = [], [], [], []
    if hp.data=="LJSpeech-1.0":
        metadata = os.path.join(hp.data, 'metadata.csv')
        for line in codecs.open(metadata, 'r', 'utf-8'):
            fname, _, sent = line.strip().split("|")
            sent = text_normalize(sent) + "E" # text normalization, E: EOS
            if len(sent) <= hp.Tx:
                sent += "P"*(hp.Tx-len(sent))
                texts.append(np.array([char2idx[char] for char in sent], np.int32).tostring())
                mels.append(os.path.join(hp.data, "mels", fname + ".npy"))
                dones.append(os.path.join(hp.data, "dones", fname + ".npy"))
                mags.append(os.path.join(hp.data, "mags", fname + ".npy"))
    else: # kate
        metadata = os.path.join(hp.data, 'text.tsv')
        for line in codecs.open(metadata, 'r', 'utf-8'):
            fname, sent, duration = line.strip().split("\t")
            if not "therese_raquin" in fname: continue
            sent = text_normalize(sent) + "E"  # text normalization, E: EOS
            if len(sent) <= hp.Tx:
                sent += "P" * (hp.Tx - len(sent))
                texts.append(np.array([char2idx[char] for char in sent], np.int32).tostring())
                mels.append(os.path.join(hp.data, "mels", fname.split("/")[-1] + ".npy"))
                dones.append(os.path.join(hp.data, "dones", fname.split("/")[-1] + ".npy"))
                mags.append(os.path.join(hp.data, "mags", fname.split("/")[-1] + ".npy"))

    return texts*1, mels*1, dones*1, mags*1

def load_test_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts = []
    for line in codecs.open('sample_sents.txt', 'r', 'utf-8'):
        sent = text_normalize(line).strip() + "E" # text normalization, E: EOS
        if len(sent) <= hp.Tx:
            sent += "P"*(hp.Tx-len(sent))
            texts.append([char2idx[char] for char in sent])
    texts = np.array(texts, np.int32)
    return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _mels, _dones, _mags = load_train_data() # bytes

        # Calc total batch count
        num_batch = len(_texts) // hp.batch_size
         
        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts)
        mels = tf.convert_to_tensor(_mels)
        dones = tf.convert_to_tensor(_dones)
        mags = tf.convert_to_tensor(_mags)
         
        # Create Queues
        text, mel, done, mag = tf.train.slice_input_producer([texts, mels, dones, mags], shuffle=True)

        # Decoding.
        text = tf.decode_raw(text, tf.int32) # (Tx,)
        mel = tf.py_func(lambda x:np.load(x), [mel], tf.float32) # (Ty, n_mels)
        done = tf.py_func(lambda x:np.load(x), [done], tf.int32) # (Ty,)
        mag = tf.py_func(lambda x:np.load(x), [mag], tf.float32) # (Ty, 1+n_fft/2)

        # Create batch queues
        texts, mels, dones, mags = tf.train.batch([text, mel, done, mag],
                                shapes=[(hp.Tx,), (hp.Ty, hp.n_mels), (hp.Ty,), (hp.Ty, 1+hp.n_fft//2)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=False)

    return texts, mels, dones, mags, num_batch