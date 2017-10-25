# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
from utils import restore_shape
from data_load import load_vocab
import tensorflow as tf
import math

def encoder(inputs, training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T_x], with dtype of int32. Encoder inputs.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Text Embedding
        embedding = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, T_x, E)

        # Encoder PreNet
        inputs = tf.layers.dense(embedding, units=hp.enc_channels, activation=None, name="encoder_prenet") # (N, T_x, 64)
        inputs = normalize(inputs, type=hp.norm_type, training=training, activation_fn=tf.nn.relu)

        # Convolution Blocks: dropout -> convolution -> glu (activation)-> residual connection -> scale
        for i in range(hp.enc_layers):
            inputs = conv_block(inputs,
                               size=hp.enc_filter_size,
                               training=training,
                               scope="encoder_conv_block")

        # Encoder PostNet : Restore the shape
        inputs = tf.layers.dense(inputs, units=hp.embed_size, activation=None, name="encoder_postnet")  # (N, T_x, E)
        keys = normalize(inputs, type=hp.norm_type, training=training, activation_fn=tf.nn.relu)

    return key, value

def decoder(inputs, memory, keys, training=True, scope="decoder1", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = tf.layers.dense(inputs, units=128, activation=tf.nn.relu,
                                 name="decoder_prenet")  # (N, T_x, c)
        weight_norm

        # Convolution Blocks: dropout -> convolution -> glu (activation)-> residual connection -> scale
        for i in range(4):
            _inputs = tf.layers.dropout(inputs, rate=hp.dropout_rate, training=training, name="dropout1")
            queries = conv1d(inputs, .., padding="causal", size=5, scope="conv1d_1")  # (N, T_y/r, 128)
            #keys # (N, T_x, E)

            # positional encoding
            ...

            # concat
            queries += positional encoding
            keys += positional encoding

            # Affine transformation
            queries = tf.layers.dense(inputs, units=128, activation=tf.nn.relu??,
                                 name="decoder_prenet")  # (N, T_x, c)
            keys = tf.layers.dense(inputs, units=128, activation=tf.nn.relu??,
                                 name="decoder_prenet")  # (N, T_x, c)

            # multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (N, T_y, T_x)



            # Activation (softmax)
            outputs = tf.nn.softmax(outputs) # <- aligments

            # Dropouts
            _inputs = tf.layers.dropout(inputs, rate=hp.dropout_rate, training=tf.convert_to_tensor(training), name="dropout1")

            #
            ## values (N, T_x, E)
            values = tf.layers.dense(values, units=128, activation=tf.nn.relu??,
                                 name="decoder_prenet")  # (N, T_x, 128)

            # multiplication
            outputs = tf.batch_matmul(outputs, values) # (N, T_y, 128)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # fc
            mel_outputs = tf.layers.dense(outputs, units=128, activation=tf.nn.relu??,
                name = "decoder_prenet")  # (N, T_x, 128)

            # fc2
            dones = tf.layers.dense(outputs, units=128, activation=tf.nn.sigmoid,
                name = "decoder_prenet")  # (N, T_x, 128)



    return outputs, alignments

def converter(inputs, training=True, scope="decoder2", reuse=None):
    '''Decoder Post-processing net = CBHG
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
      training: Whether or not the layer is in training mode.  
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted linear spectrogram tensor with shape of [N, T_y, 1+n_fft//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        Restore shape
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])
        inputs = restore_shape(inputs) # (N, T_y, n_mels)

        # Conv1D bank
        # Convolution Blocks: dropout -> convolution -> glu (activation)-> residual connection -> scale
        for i in range(5):
            inputs = conv1d(inputs, hp.enc_channels * 2, size=5, scope="conv1d_1")  # (N, T_y, E)

        # FC
        outputs = tf.layers.dense(outputs, units=1+2//n_fft, activation=tf.nn.relu??,
        name = "decoder_prenet")

    return outputs
