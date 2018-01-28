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

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode in ("train", "eval"):
        # Parse
        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(hp.data, 'transcript.csv')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        if mode=="train":
            lines = lines[1:]
        else:
            lines = lines[:1]

        for line in lines:
            fname, _, text = line.strip().split("|")

            fpath = os.path.join(hp.data, "wavs", fname + ".wav")
            fpaths.append(fpath)

            text = text_normalize(text) + "E"  # E: EOS
            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())

        return fpaths, text_lengths, texts

    else:
        # Parse
        lines = codecs.open('harvard_sentences.txt', 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
        lengths = [len(sent) for sent in sents]
        maxlen = sorted(lengths, reverse=True)[0]
        texts = np.zeros((len(sents), maxlen), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)
        mel, mag = tf.py_func(get_spectrograms, [fpath], [tf.float32, tf.float32])  # (None, n_mels)

        # Padding
        text = tf.pad(text, ((0, hp.Tx),))[:hp.Tx]  # (Tx,)
        mel = tf.pad(mel, ((0, hp.Ty), (0, 0)))[:hp.Ty]  # (Ty, n_mels)
        mag = tf.pad(mag, ((0, hp.Ty), (0, 0)))[:hp.Ty]  # (Ty, 1+n_fft/2)

        # Reduction
        mel = tf.reshape(mel, (hp.Ty//hp.r, -1))  # (Ty/r, n_mels*r)
        done = tf.ones_like(mel[:, 0], dtype=tf.int32)  # (Ty/r,)

        # Add shape information
        text.set_shape((hp.Tx,))
        mel.set_shape((hp.Ty//hp.r, hp.n_mels*hp.r))
        done.set_shape((hp.Ty//hp.r, ))
        mag.set_shape((hp.Ty, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, dones, mags) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, done, mag],
                                            batch_size=hp.batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=16,
                                            capacity=hp.batch_size * 4,
                                            dynamic_pad=True)

    return texts, mels, dones, mags, num_batch

