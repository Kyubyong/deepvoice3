# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''
from __future__ import print_function

import numpy as np
import librosa
import copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal

from hyperparams import Hyperparams as hp


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = librosa.db_to_amplitude(mag)
    # print(np.max(mag), np.min(mag), mag.shape)
    # (1025, 812, 16)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignments, gs):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    fig, axes = plt.subplots(nrows=len(alignments), ncols=1, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        # i=0
        im = ax.imshow(alignments[i])
        ax.axis('off')
        ax.set_title("Layer {}".format(i))

    fig.subplots_adjust(right=0.8, hspace=0.4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(hp.logdir, gs), format='png')

# def cross_entropy_loss(predictions, labels, num_classes):
#     def _softmax(X):
#         exps = np.exp(X) - np.max(X)
#         return exps / np.sum(exps, axis=-1, keepdims=True)
#
#     def _onehot(X, num_classes):
#         n, t = X.shape
#         ret = np.zeros([n, t, num_classes], np.int32)
#         for i in range(n):
#             for j in range(t):
#                 ret[:, :, X[i, j]] = 1
#         return ret
#
#     y = _onehot(labels, num_classes)
#     p = np.clip(_softmax(predictions), 1e-7, 1 - 1e-7)
#     return np.where(y == 1, -np.log(p), -np.log(1 - p))