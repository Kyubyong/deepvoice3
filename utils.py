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

from hyperparams import Hyperparams as hp


def spectrogram2wav(spectrogram):
    '''Convert spectrogram into a waveform using Griffin-lim's raw.
    '''
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignment, gs):
    """
    Plots the alignment

    alignment: (numpy) matrix of shape (encoder_steps,decoder_steps)

    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment, cmap='hot', interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.xlabel('Decoder timestep')
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig('{}/alignment_{}.png'.format(hp.logdir, gs), format='png')

# import codecs
# import re
# import csv
# import os
#
# def load_vocab():
#     vocab = "EG abcdefghijklmnopqrstuvwxyz'"  # E: Empty. ignore G
#     char2idx = {char: idx for idx, char in enumerate(vocab)}
#     idx2char = {idx: char for idx, char in enumerate(vocab)}
#     return char2idx, idx2char
#
# def create_train_data():
#     # Load vocabulary
#     char2idx, idx2char = load_vocab()
#
#     texts, sound_files = [], []
#     total_duration = 0
#     if hp.data == "WEB":
#         reader = csv.reader(codecs.open(os.path.join(hp.data, "text.csv"), 'rb', 'utf-8'))
#         for row in reader:
#             sound_fname, text, duration = row
#             sound_file = os.path.join(hp.data, sound_fname) + ".wav"
#             text = re.sub(r"[^ a-z']", "", text.strip().lower())
#             duration = float(duration)
#
#             if duration < hp.max_duration:
#                 texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
#                 sound_files.append(sound_file)
#                 total_duration += duration
#     elif hp.data == "LJ":
#         reader = csv.reader(codecs.open("LJSpeech-1.0/metadata.csv", 'rb', 'utf-8'))
#         for line in codecs.open("LJSpeech-1.0/metadata.csv", 'r', 'utf-8'):
#             sound_fname ,_ ,text = line.split('|')
#             sound_file = "LJSpeech-1.0/wavs/" + sound_fname + ".wav"
#             text = re.sub(r"[^ a-z']", "", text.strip().lower())
#             duration = float(len(text)/25.)
#
#             if duration < hp.max_duration:
#                 texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
#                 sound_files.append(sound_file)
#
#     else: # Kate
#         for line in codecs.open(hp.data + "/text.tsv", 'r', 'utf-8'):
#             sound_fname, text, duration = line.split("\t")
#             sound_file = hp.data + "/" + sound_fname + ".wav"
#             text = re.sub(r"[^ a-z']", "", text.strip().lower())
#             duration = float(duration)
#
#             if duration < hp.max_duration:
#                 texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
#                 sound_files.append(sound_file)
#
#     return texts, sound_files, total_duration
#
# def load_train_data():
#     """We train on the whole data but the last num_samples."""
#     texts, sound_files, total_duration = create_train_data()
#     if hp.sanity_check:  # We use a single mini-batch for training to overfit it.
#         texts, sound_files = texts[:hp.batch_size] * 1000, sound_files[:hp.batch_size] * 1000
#     else:
#         texts, sound_files = texts[:-hp.num_samples], sound_files[:-hp.num_samples]
#     print("total_duration = ", total_duration/3600, "hours")
#     return texts, sound_files
#
# def load_eval_data():
#     """We evaluate on the last num_samples."""
#     texts, _, _ = create_train_data()
#     if hp.sanity_check:  # We generate samples for the same texts as the ones we've used for training.
#         texts = texts[:hp.batch_size]
#     else:
#         texts = texts[-hp.num_samples:]
#
#     X = np.zeros(shape=[len(texts), hp.max_len], dtype=np.int32)
#     for i, text in enumerate(texts):
#         _text = np.fromstring(text, np.int32)  # byte to int
#         X[i, :len(_text)] = _text
#
#     return X
#
#
# mels, mags = [], []
# _, sound_files = load_train_data()
# for sound_f in sound_files[:100]:
#     print(sound_f)
#     mel, mag = get_spectrograms(sound_f)
#     mel = np.log(mel+ 1e-8)
#     mag = np.log(mag+ 1e-8)
#     mels.extend(list(mel.flatten()))
#     mags.extend(list(mag.flatten()))
#
# print(np.mean(np.array(mels)))
# print(np.std(np.array(mels)))
# print(np.mean(np.array(mags)))
# print(np.std(np.array(mags)))
#
