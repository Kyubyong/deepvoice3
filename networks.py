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
        tensor = tf.layers.dense(embedding, units=hp.enc_channels, activation=None, name="encoder_prenet") # (N, T_x, 64)
        tensor = normalize(tensor, type=hp.norm_type, training=training, activation_fn=tf.nn.relu)

        # Convolution Blocks: dropout -> convolution -> glu (activation)-> residual connection -> scale
        for i in range(hp.enc_layers):
            tensor = conv_block(tensor,
                               size=hp.enc_filter_size,
                               training=training,
                               scope="encoder_conv_block_{}".format(i))

        # Encoder PostNet : Restore the shape
        tensor = tf.layers.dense(tensor, units=hp.embed_size, activation=None, name="encoder_postnet")  # (N, T_x, E)
        keys = normalize(tensor, type=hp.norm_type, training=training, activation_fn=tf.nn.relu)
        vals = math.sqrt(0.5) * (keys + embedding)

    return keys, vals

def decoder(inputs, keys, vals, training=True, scope="decoder", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Shifted log melspectrogram of sound files.
      keys: A 3d tensor with shape of [N, T_x, E].
      vals: A 3d tensor with shape of [N, T_x, E].
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder PreNet
        for i in range(hp.dec_layers):
            inputs = tf.layers.dropout(inputs, rate=hp.dropout_rate, training=training)
            inputs = tf.layers.dense(inputs, units=hp.dec_affine_size[0], activation=None)  # (N, T_y//r, c0)
            inputs = normalize(inputs, type=hp.norm_type, training=training, activation_fn=tf.nn.relu)

        for i in range(hp.dec_layers):
            # Causal Convolution Block
            queries = conv_block(inputs,
                                size=hp.dec_filter_size,
                                training=training,
                                padding="CAUSAL",
                                scope="decoder_conv_block_{}".format(i))
            # positional encoding
            # queries += ...
            # keys += ...

            # Attention Block
            queries = tf.layers.dense(queries, units=hp.dec_affine_size[1], activation=tf.nn.relu)  # (N, T_y//r, c1)
            keys = tf.layers.dense(keys, units=hp.dec_affine_size[1], activation=tf.nn.relu) # (N, T_x, c1)
            vals = tf.layers.dense(vals, units=hp.dec_affine_size[1], activation=tf.nn.relu)  # (N, T_x, c1)

            attention_weights = tf.matmul(queries, keys, transpose_b=True)  # (N, T_y//r, T_x)
            alignments = tf.nn.softmax(attention_weights)
            dropout = tf.layers.dropout(alignments, rate=hp.dropout_rate, training=training)

            tensor = tf.matmul(dropout, vals) # (N, T_y//r, c1)
            tensor /= math.sqrt(hp.dec_affine_size[1])

            tensor = normalize(tensor, type=hp.norm_type, training=training, activation_fn=None)
            inputs = tensor + queries

        # fc1
        mels = tf.layers.dense(inputs, units=hp.n_mels*hp.r, activation=None) # (N, T_y/r, n_mels*r)

        # fc2
        dones = tf.layers.dense(inputs, units=2, activation=None) # (N, T_y//r, 2)
    return mels, dones, alignments

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
