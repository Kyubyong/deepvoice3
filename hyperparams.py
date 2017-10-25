# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
import math

def get_T_y(duration, sr, win_length, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T

class Hyperparams:
    '''Hyper parameters'''
    # mode
    sanity_check = False

    # signal processing
    sr = 22050 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    mag_mean = -4.
    mag_std = 3.
    mel_mean = -5
    mel_std = 2.5

    # Model
    norm_type = "bn"
    ## Enocder
    vocab_size = 30 # [PE a-z'] == V
    embed_size = 256 # == E
    enc_layers = 7
    enc_filter_size = 5
    enc_channels = 64 # == c

    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    feed_last_frames = True
    r = 4 # Reduction factor
    dropout_rate = .5

    # data
    data = 'LJSpeech-1.0'
    max_duration = 10.10 # seconds
    T_x = 150 # characters. maximum length of text.
    T_y = int(get_T_y(max_duration, sr, win_length, hop_length, r)) # Maximum length of sound (frames)

    # training scheme
    lr = 0.001
    logdir = "logdir/trial0"
    sampledir = 'samples/trial0'
    batch_size = 16
    num_epochs = 10000 if not sanity_check else 60 # Paper => 2M global steps!
    num_samples = 32




