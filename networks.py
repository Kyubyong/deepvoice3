# -*- coding: utf-8 -*-
#/usr/bin/python2
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
      inputs: A 2d tensor with shape of [N, T_x], with dtype of int32. Encoder inputs.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, e).
    '''
    masks = tf.sign(tf.abs(inputs))  # (N, T_x)
    with tf.variable_scope(scope, reuse=reuse):
        # Text Embedding
        embedding = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, T_x, e)

        # Encoder PreNet
        tensor = fc_block(embedding,
                          num_units=hp.enc_channels,
                          dropout_rate=0,
                          norm_type=hp.norm_type,
                          activation_fn=None,
                          training=training,
                          scope="prenet_fc_block") # (N, T_x, c)

        # Convolution Blocks
        for i in range(hp.enc_layers):
            tensor = conv_block(tensor,
                                size=hp.enc_filter_size,
                                norm_type=hp.norm_type,
                                activation_fn=glu,
                                training=training,
                                scope="encoder_conv_block_{}".format(i)) # (N, T_x, c)

        # Encoder PostNet
        keys = fc_block(tensor,
                        num_units=hp.embed_size,
                        dropout_rate=0,
                        norm_type=None,
                        activation_fn=None,
                        training=training,
                        scope="postnet_fc_block") # (N, T_x, e)
        vals = tf.sqrt(0.5) * (keys + embedding) # (N, T_x, e)

    return keys, vals, masks

def decoder(inputs,
            keys,
            vals,
            masks,
            prev_max_attentions=None,
            training=True,
            scope="decoder",
            reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels]. Shifted log melspectrogram of sound files.
      keys: A 3d tensor with shape of [N, T_x, e].
      vals: A 3d tensor with shape of [N, T_x, e].
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder PreNet. inputs:(N, T_y/r, e)
        for i in range(hp.dec_layers):
            inputs = fc_block(inputs,
                              num_units=hp.embed_size,
                              dropout_rate=0 if i==0 else hp.dropout_rate,
                              norm_type=hp.norm_type,
                              activation_fn=tf.nn.relu,
                              training=training,
                              scope="prenet_fc_block_{}".format(i))

        for i in range(hp.dec_layers):
            # Causal Convolution Block. queries: (N, T_y/r, e)
            queries = conv_block(inputs,
                                 size=hp.dec_filter_size,
                                 padding="CAUSAL",
                                 norm_type=hp.norm_type,
                                 activation_fn=glu,
                                 training=training,
                                 scope="decoder_conv_block_{}".format(i))

            # Attention Block. tensor: (N, T_y/r, e), alignments: (N, T_y, T_x)
            tensor, alignments, max_attentions = attention_block(queries,
                                                                 keys,
                                                                 vals,
                                                                 masks,
                                                                 num_units=hp.attention_size,
                                                                 dropout_rate=hp.dropout_rate,
                                                                 prev_max_attentions=prev_max_attentions,
                                                                 norm_type=hp.norm_type,
                                                                 activation_fn=None,
                                                                 training=training,
                                                                 scope="attention_block_{}".format(i))

            inputs = tensor + queries

        decoder_output = inputs

        # Readout layers: mel_output: (N, T_y/r, n_mels*r)
        mel_output = fc_block(inputs,
                        num_units=hp.n_mels*hp.r*2,
                        dropout_rate=0,
                        norm_type=None,
                        activation_fn=None,
                        training=training,
                        scope="mels")  # (N, T_y/r, n_mels*r*2)
        A, B = tf.split(mel_output, 2, -1)
        mel_output = A*tf.nn.sigmoid(B)

        ## done_output: # (N, T_y/r, 2)
        done_output = fc_block(inputs,
                         num_units=2,
                         dropout_rate=0,
                         norm_type=None,
                         activation_fn=None,
                         training=training,
                         scope="dones")
    return mel_output, done_output, decoder_output, alignments, max_attentions

def converter(inputs, training=True, scope="converter", reuse=None):
    '''Converter
    Args:
      inputs: A 3d tensor with shape of [N, T_y, e/r]. Activations of the reshaped outputs of the decoder.
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
                                 norm_type=hp.norm_type,
                                 activation_fn=glu,
                                 training=training,
                                 scope="converter_conv_block_{}".format(i))  # (N, T_y/r, d)

        # Readout layer. mag_output: (N, T_y, n_fft/2+1)
        mag_output = fc_block(inputs,
                       num_units=hp.n_fft//2+1,
                       dropout_rate=0,
                       norm_type=None,
                       activation_fn=None,
                       training=training,
                       scope="mag")

    return mag_output
