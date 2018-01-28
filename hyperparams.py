# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''
class Hyperparams:
    '''Hyper parameters'''
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence

    # data
    data = "C:/LJSpeech-1.0"
    Tx = 145
    Ty = 800

    # signal processing
    sr = 22050 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    ## Enocder
    embed_size = 256  # == e
    enc_layers = 7
    enc_filter_size = 3
    enc_channels = 64  # == c
    ## Decoder
    dec_layers = 1
    dec_filter_size = 5
    attention_size = 128 * 2  # == a
    ## Converter
    converter_layers = 5
    converter_filter_size = 5
    converter_channels = 256  # == v
    attention_win_size = 3

    r = 4 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .05

    # training scheme
    lr = 0.0005 # Paper => Exponential decay
    logdir = "logdir/05"
    sampledir = 'samples'
    batch_size = 32
    max_grad_norm = 100.
    max_grad_val = 1.
    num_iterations = 500000




