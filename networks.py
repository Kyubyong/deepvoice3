# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def encoder(inputs, training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, Tx], with dtype of int32. Encoder inputs.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, Tx, e).
    '''
    masks = tf.sign(tf.abs(inputs))  # (N, Tx)
    with tf.variable_scope(scope, reuse=reuse):
        # Text Embedding
        embedding = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, Tx, e)

        print("# Encoder PreNet")
        tensor = fc_block(embedding,
                          num_units=hp.enc_channels,
                          dropout_rate=0,
                          activation_fn=None,
                          training=training,
                          scope="prenet_fc_block")  # (N, Tx, c)

        print("# Convolution Blocks")
        for i in range(hp.enc_layers):
            tensor = conv_block(tensor,
                                size=hp.enc_filter_size,
                                training=training,
                                scope="encoder_conv_block_{}".format(i))  # (N, Tx, c)

        print("# Encoder PostNet")
        keys = fc_block(tensor,
                        num_units=hp.embed_size,
                        dropout_rate=0,
                        activation_fn=None,
                        training=training,
                        scope="postnet_fc_block")  # (N, Tx, e)
        vals = tf.sqrt(0.5) * (keys + embedding)  # (N, Tx, e)

    return keys, vals


def decoder(inputs,
            keys,
            vals,
            prev_max_attentions=None,
            training=True,
            scope="decoder",
            reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, Ty, n_mels]. Shifted log melspectrogram of sound files.
      keys: A 3d tensor with shape of [N, Tx, e].
      vals: A 3d tensor with shape of [N, Tx, e].
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder PreNet. inputs:(N, Ty, e)
        for i in range(hp.dec_layers):
            inputs = fc_block(inputs,
                              num_units=hp.embed_size,
                              dropout_rate=0 if i == 0 else hp.dropout_rate,
                              activation_fn=tf.nn.relu,
                              training=training,
                              scope="prenet_fc_block_{}".format(i))

        alignments = []
        for i in range(hp.dec_layers):
            # Causal Convolution Block. queries: (N, Ty, e)
            queries = conv_block(inputs,
                                 size=hp.dec_filter_size,
                                 padding="CAUSAL",
                                 training=training,
                                 scope="decoder_conv_block_{}".format(i))

            # Attention Block. tensor: (N, Ty, e), alignment: (Tx, Ty)
            tensor, alignment, max_attentions = attention_block(queries,
                                                                 keys,
                                                                 vals,
                                                                 num_units=hp.attention_size,
                                                                 dropout_rate=hp.dropout_rate,
                                                                 prev_max_attentions=prev_max_attentions,
                                                                 training=training,
                                                                 scope="attention_block_{}".format(i))
            alignments.append(alignment)
            inputs = (tensor + queries) * tf.sqrt(0.5)

        decoder_output = inputs # (N, Ty, e)

        # Readout layers. mel_output: (N, Ty, n_mels*r)
        mel_output = fc_block(inputs,
                              num_units=hp.n_mels * hp.r * 2,
                              dropout_rate=0,
                              activation_fn=None,
                              training=training,
                              scope="mels")
        mel_output = glu(mel_output)

        ## done_output: # (N, Ty, 2)
        done_output = fc_block(inputs,
                               num_units=2,
                               dropout_rate=0,
                               activation_fn=None,
                               training=training,
                               scope="dones")
    return mel_output, done_output, decoder_output, alignments, max_attentions


def converter(inputs, training=True, scope="converter", reuse=None):
    '''Converter
    Args:
      inputs: A 3d tensor with shape of [N, Ty, e/r]. Activations of the reshaped outputs of the decoder.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(hp.converter_layers):
            inputs = conv_block(inputs,
                                size=hp.converter_filter_size,
                                padding="SAME",
                                training=training,
                                scope="converter_conv_block_{}".format(i))  # (N, Ty/r, d)

        # Readout layer. mag_output: (N, Ty, n_fft/2+1)
        mag_output = fc_block(inputs,
                              num_units=hp.n_fft // 2 + 1,
                              dropout_rate=0,
                              activation_fn=None,
                              training=training,
                              scope="mag")

    return mag_output
