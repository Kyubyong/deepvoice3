#/usr/bin/python2
# -*- coding: utf-8 -*-

'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

import numpy as np
import librosa

from hyperparams import Hyperparams as hp
import glob
import os
import tqdm


def get_spectrograms(sound_file):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                     n_fft=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear) # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag**2)  # (n_mels, t)

    # Transpose
    mel = mel.T.astype(np.float32) # (T, n_mels)
    mag = mag.T.astype(np.float32) # (T, 1+n_fft//2)

    # Sequence length
    dones = np.ones_like(mel[:, 0])

    # Padding
    mel = np.pad(mel, ((0, hp.T_y - len(mel)), (0, 0)), mode="constant")[:hp.T_y]
    mag = np.pad(mag, ((0, hp.T_y - len(mag)), (0, 0)), mode="constant")[:hp.T_y]
    dones = np.pad(dones, ((0, hp.T_y - len(dones))), mode="constant")[:hp.T_y]

    # Log
    mel = np.log10(mel + 1e-8)
    mag = np.log10(mag + 1e-8)

    # Normalize
    mel = (mel - hp.mel_mean) / hp.mel_std
    mag = (mag - hp.mag_mean) / hp.mag_std

    # Reduce frames for net1
    mel = np.reshape(mel, (-1, hp.n_mels*hp.r))
    dones = np.reshape(dones, (-1, hp.r))
    dones = np.equal(np.sum(dones, -1), 0).astype(np.int32) # 1 for done, 0 for undone.

    return mel, dones, mag # (T/r, n_mels*r), (T/r,), (T, 1+n_fft/2)

if __name__ == "__main__":
    wav_folder = os.path.join(hp.data, 'wavs')
    mel_folder = os.path.join(hp.data, 'mels')
    dones_folder = os.path.join(hp.data, 'dones')
    mag_folder = os.path.join(hp.data, 'mags')

    for folder in (mel_folder, dones_folder, mag_folder):
        if not os.path.exists(folder): os.mkdir(folder)

    files = glob.glob(os.path.join(wav_folder, "*"))
    for f in tqdm.tqdm(files):
        fname = os.path.basename(f)
        mel, dones, mag = get_spectrograms(f)  # (n_mels, T), (1+n_fft/2, T) float32
        np.save(os.path.join(mel_folder, fname.replace(".wav", ".npy")), mel)
        np.save(os.path.join(dones_folder, fname.replace(".wav", ".npy")), dones)
        np.save(os.path.join(mag_folder, fname.replace(".wav", ".npy")), mag)
