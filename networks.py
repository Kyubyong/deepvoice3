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
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("text_embedding"):
            embedding = embed(inputs, len(hp.vocab), hp.embed_size)  # (N, Tx, e)

        with tf.variable_scope("encoder_prenet"):
            tensor = fc_block(embedding, hp.enc_channels, training=training)  # (N, Tx, c)

        with tf.variable_scope("encoder_conv"):
            for i in range(hp.enc_layers):
                tensor = conv_block(tensor,
                                     size=hp.enc_filter_size,
                                     rate=1,
                                     training=training,
                                     scope="encoder_conv_{}".format(i))  # (N, Tx, c)

        with tf.variable_scope("encoder_postnet"):
            keys = fc_block(tensor, hp.embed_size, training=training)  # (N, Tx, e)
            vals = tf.sqrt(0.5) * (keys + embedding)  # (N, Tx, e)

    return keys, vals


def decoder(inputs,
            keys,
            vals,
            prev_max_attentions_li=None,
            training=True,
            scope="decoder",
            reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, Ty/r, n_mels]. Shifted log melspectrogram of sound files.
      keys: A 3d tensor with shape of [N, Tx, e].
      vals: A 3d tensor with shape of [N, Tx, e].
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("decoder_prenet"):
            for i in range(hp.dec_layers):
                inputs = fc_block(inputs,
                                  num_units=hp.embed_size,
                                  dropout_rate=0 if i == 0 else hp.dropout_rate,
                                  activation_fn=tf.nn.relu,
                                  training=training,
                                  scope="decoder_prenet_{}".format(i))  # (N, Ty/r, a)

        with tf.variable_scope("decoder_conv_att"):
            with tf.name_scope("positional_encoding"):
                query_pe = positional_encoding(inputs[:, :, 0],
                                               num_units=hp.embed_size,
                                               position_rate=1.,
                                               zero_pad=False,
                                               scale=True)  # (N, Ty/r, e)
                key_pe = positional_encoding(keys[:, :, 0],
                                             num_units=hp.embed_size,
                                             position_rate=(hp.Ty // hp.r) / hp.Tx,
                                             zero_pad=False,
                                             scale=True)  # (N, Tx, e)

            alignments_li, max_attentions_li = [], []
            for i in range(hp.dec_layers):
                queries = conv_block(inputs,
                                     size=hp.dec_filter_size,
                                     rate=1,
                                     padding="CAUSAL",
                                     training=training,
                                     scope="decoder_conv_block_{}".format(i))  # (N, Ty/r, a)

                # residual connection
                queries += query_pe
                keys += key_pe

                # Attention Block.
                # tensor: (N, Ty/r, e)
                # alignments: (N, Ty/r, Tx)
                # max_attentions: (N, Ty/r)
                tensor, alignments, max_attentions = attention_block(queries,
                                                                     keys,
                                                                     vals,
                                                                     dropout_rate=hp.dropout_rate,
                                                                     prev_max_attentions=prev_max_attentions_li[i],
                                                                     mononotic_attention=(not training),
                                                                     training=training,
                                                                     scope="attention_block_{}".format(i))

                inputs = (tensor + queries) * tf.sqrt(0.5)

                alignments_li.append(alignments)
                max_attentions_li.append(max_attentions)

        decoder_output = inputs

        with tf.variable_scope("mel_logits"):
            mel_logits = fc_block(decoder_output, hp.n_mels * hp.r, training=training)  # (N, Ty/r, n_mels*r)

        with tf.variable_scope("done_output"):
            done_output = fc_block(inputs, 2, training=training)  # (N, Ty/r, 2)

    return mel_logits, done_output, decoder_output, alignments_li, max_attentions_li


def converter(inputs, training=True, scope="converter", reuse=None):
    '''Converter
    Args:
      inputs: A 3d tensor with shape of [N, Ty, v]. Activations of the reshaped outputs of the decoder.
      training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("converter_conv"):
            for i in range(hp.converter_layers):
                inputs = conv_block(inputs,
                                     size=hp.converter_filter_size,
                                     rate=1,
                                     padding="SAME",
                                     training=training,
                                     scope="converter_conv_{}".format(i))  # (N, Ty/r, d)
                # inputs = (inputs + outputs) * tf.sqrt(0.5)

        with tf.variable_scope("mag_logits"):
            mag_logits = fc_block(inputs, hp.n_fft // 2 + 1, training=training)  # (N, Ty, n_fft/2+1)

    return mag_logits
