# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor. 
    
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    return outputs
 
def glu(inputs):
    '''Gated linear unit
    Args:
      inputs: A tensor of even dimensions. (N, Tx, 2c)

    Returns:
      outputs: A tensor of the same shape and dtype as inputs.
    '''
    a, b = tf.split(inputs, 2, -1)  # (N, Tx, c) * 2
    outputs = a * tf.nn.sigmoid(b)
    return outputs

def conv_block(inputs,
               num_units=None,
               size=5,
               rate=1,
               padding="SAME",
               dropout_rate=0,
               training=False,
               scope="conv_block",
               reuse=None):
    '''Convolution block.
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      size: An int. Filter size.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      norm_type: A string. See `normalize`.
      activation_fn: A string. Activation function.
      training: A boolean. Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A tensor of the same shape and dtype as inputs.
    '''
    in_dim = inputs.get_shape().as_list()[-1]
    if num_units is None: num_units = in_dim

    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "VALID"

        V = tf.get_variable('V',
                            shape=[size, in_dim, num_units*2],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=(4.*(1.-dropout_rate)))) # (width, in_dim, out_dim)
        g = tf.get_variable('g',
                            dtype=tf.float32,
                            initializer=tf.norm(V.initialized_value(), axis=(0, 1), keep_dims=True)
                            )
        b = tf.get_variable('b',
                            shape=(num_units*2,),
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer)

        V_norm = tf.nn.l2_normalize(V, [0, 1])  # (width, in_dim, out_dim)
        W = V_norm * g

        outputs = tf.nn.convolution(inputs, W, padding, dilation_rate=[rate]) + b
        outputs = glu(outputs)

    return outputs

def fc_block(inputs,
             num_units,
             dropout_rate=0,
             activation_fn=None,
             training=False,
             scope="fc_block",
             reuse=None):
    '''Fully connected layer block.
        Args:
          inputs: A 3-D tensor with shape of [batch, time, depth].
          num_units: An int. Output dimensionality.
          dropout_rate: A float of [0, 1]. Dropout rate.
          norm_type: A string. See `normalize`.
          activation_fn: A string. Activation function.
          training: A boolean. Whether or not the layer is in training mode.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          A tensor with shape of [batch, time, num_units].
    '''
    _, T, in_dim = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

        # Transformation
        V = tf.get_variable('V',
                            shape=[in_dim, num_units],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(
                                factor=(1. - dropout_rate))) # (in_dim, num_units)
        g = tf.get_variable('g',
                            dtype=tf.float32,
                            initializer=tf.norm(V.initialized_value(), axis=0, keep_dims=True))
        b = tf.get_variable('b', shape=(num_units), dtype=tf.float32, initializer=tf.zeros_initializer)

        V_norm = tf.nn.l2_normalize(V, [0]) # (in_dim, num_units)
        W = V_norm * g

        outputs = tf.matmul(tf.reshape(inputs, (-1, in_dim)), W) + b # (N*T, num_units)
        outputs = tf.reshape(outputs, (-1, T, num_units)) # (N, T, num_units)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def positional_encoding(inputs,
                        num_units,
                        position_rate=1.,
                        zero_pad=False,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      position_rate: A float. Average slope of the line in the attention distribution
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos*position_rate / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs *= num_units**0.5

        return outputs

def attention_block(queries,
                    keys,
                    vals,
                    dropout_rate=0,
                    prev_max_attentions=None,
                    training=False,
                    mononotic_attention=False,
                    scope="attention_block",
                    reuse=None):
    '''Attention block.
     Args:
       queries: A 3-D tensor with shape of [batch, Ty//r, e].
       keys: A 3-D tensor with shape of [batch, Tx, e].
       vals: A 3-D tensor with shape of [batch, Tx, e].
       num_units: An int. Attention size.
       dropout_rate: A float of [0, 1]. Dropout rate.
       norm_type: A string. See `normalize`.
       activation_fn: A string. Activation function.
       training: A boolean. Whether or not the layer is in training mode.
       scope: Optional scope for `variable_scope`.
       reuse: Boolean, whether to reuse the weights of a previous layer
         by the same name.
    '''
    _keys = keys
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("query_proj"):
            queries = fc_block(queries, hp.attention_size, training=training) # (N, Ty/r, a)

        with tf.variable_scope("key_proj"):
            keys = fc_block(keys, hp.attention_size, training=training) # (N, Tx, a)

        with tf.variable_scope("value_proj"):
            vals = fc_block(vals, hp.attention_size, training=training) # (N, Tx, a)

        with tf.variable_scope("alignments"):
            attention_weights = tf.matmul(queries, keys, transpose_b=True)  # (N, Ty/r, Tx)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, Tx)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (N, Ty/r, Tx)

            paddings = tf.ones_like(attention_weights) * (-2 ** 32 + 1)
            attention_weights = tf.where(tf.equal(key_masks, 0), paddings, attention_weights)  # (N, Ty/r, Tx)

            _, Ty, Tx = attention_weights.get_shape().as_list()  # Ty=Ty/r, Tx = Tx
            if mononotic_attention: # for inference
                key_masks = tf.sequence_mask(prev_max_attentions, Tx)
                reverse_masks = tf.sequence_mask(Tx - hp.attention_win_size - prev_max_attentions, Tx)[:, ::-1]
                masks = tf.logical_or(key_masks, reverse_masks)
                masks = tf.tile(tf.expand_dims(masks, 1), [1, Ty, 1])
                paddings = tf.ones_like(attention_weights) * (-2 ** 32 + 1)  # (N, Ty/r, Tx)
                attention_weights = tf.where(tf.equal(masks, False), attention_weights, paddings)
            alignments = tf.nn.softmax(attention_weights)
            max_attentions = tf.argmax(alignments, -1) # (N, Ty/r)

        with tf.variable_scope("context"):
            ctx = tf.layers.dropout(alignments, rate=dropout_rate, training=training)
            ctx = tf.matmul(ctx, vals)  # (N, Ty/r, a)
            ctx *= tf.rsqrt(tf.to_float(Tx))

        # Restore shape for residual connection
        tensor = fc_block(ctx, hp.embed_size, training=training)  # (N, Tx, e)

        # returns the alignment of the first one
        alignments = tf.transpose(alignments[0])[::-1, :]  # (Tx, Ty)

    return tensor, alignments, max_attentions