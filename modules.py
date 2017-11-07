# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''

from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import time

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
    A, B = tf.split(inputs, 2, -1)
    return A * tf.nn.sigmoid(B)


def conv_block(inputs,
               size=5,
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
    out_dim = in_dim * 2
    _inputs = inputs
    print("START")
    START_TIME = time.time()
    print(START_TIME)
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = size - 1  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "VALID"

        print("# data based initialization of parameters")
        print(time.time()-START_TIME)
        V = tf.get_variable('V',
                            shape=[size] + [in_dim, out_dim],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=4.),
                            trainable=True) # (W, in_dim, out_dim)
        V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1]) # (W, in_dim, in_dim)
        x_init = tf.nn.conv1d(inputs, V_norm, 1, padding) # (N, W, in_dim)
        m_init, v_init = tf.nn.moments(x_init, [0, 1]) # (in_dim,)
        scale_init = 1. / tf.sqrt(v_init + 1e-8)
        g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
        b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init, trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1]) # (W, in_dim, in_dim)
        x = tf.nn.bias_add(tf.nn.conv1d(inputs, W, 1, padding), b) # (N, T, in_dim)

        print("# apply nonlinearity")
        print(time.time() - START_TIME)
        x = glu(x)
        print("apply nonlinearity Done")
        print(time.time() - START_TIME)


        # residual connection
        x += _inputs
        x *= tf.sqrt(0.5)  # scale
    return x

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
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

        # Transformation
        V = tf.get_variable('V',
                            shape=[inputs.get_shape()[0], inputs.get_shape()[-1], num_units],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1. - dropout_rate),
                            trainable=True) # (N, in_dim, num_units)
        V_norm = tf.nn.l2_normalize(V.initialized_value(), [1]) # (N, in_dim, num_units)
        x_init = tf.matmul(inputs, V_norm)  # (N, T, num_units)
        m_init, v_init = tf.nn.moments(x_init, [1]) # (N, num_units)
        scale_init = 1. / tf.sqrt(v_init + 1e-10)
        g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
        b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init, trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(inputs, V) # (N, T, num_unit)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [1])) # (N, num_units)
        x = tf.expand_dims(scaler, 1) * x + tf.expand_dims(b, 1)

        # apply nonlinearity
        if activation_fn is not None:
            x = activation_fn(x)
    return x

def positional_encoding(inputs,
                        num_units,
                        position_rate=1.,
                        zero_pad=True,
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
            [pos * position_rate / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the sine to even columns and cosine to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs *= num_units ** 0.5

        return outputs


def attention_block(queries,
                    keys,
                    vals,
                    num_units,
                    dropout_rate=0,
                    prev_max_attentions=None,
                    training=False,
                    scope="attention_block",
                    reuse=None):
    '''Attention block.
     Args:
       queries: A 3-D tensor with shape of [batch, Ty, e].
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
        if hp.sinusoid: # positional encoding as in paper
            queries += positional_encoding(queries[:, :, 0],
                                           num_units=hp.embed_size,
                                           position_rate=1.,
                                           zero_pad=False,
                                           scale=True,
                                           scope='query_pe')  # (N, Ty, e)
            keys += positional_encoding(keys[:, :, 0],
                                        num_units=hp.embed_size,
                                        position_rate=hp.Ty/hp.Tx,
                                        zero_pad=False,
                                        scale=True,
                                        scope="key_pe")  # (N, Tx, e)
        else: # positional embedding as an alternative
            queries += embed(tf.tile(tf.expand_dims(tf.range(hp.Ty), 0), [hp.batch_size, 1]),
                                  vocab_size=hp.Ty,
                                  num_units=hp.embed_size,
                                  zero_pad=False,
                                  scope="query_pe")
            keys += embed(tf.tile(tf.expand_dims(tf.range(hp.Tx), 0), [hp.batch_size, 1]),
                                  vocab_size=hp.Tx,
                                  num_units=hp.embed_size,
                                  zero_pad=False,
                                  scope="key_pe")

        # Query Projection: (N, Ty, a)
        with tf.variable_scope("query_proj"):
            W1 = tf.get_variable("W1", shape=(queries.get_shape()[0], queries.get_shape()[-1], num_units),
                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.))
            b1 = tf.get_variable("b1", shape=(num_units), initializer=tf.zeros_initializer())
            queries = tf.matmul(queries, W1) + b1

        # Key Projection: (N, Tx, a)
        with tf.variable_scope("key_proj"):
            W2 = tf.get_variable("W2", initializer=W1.initialized_value())
            b2 = tf.get_variable("b2", shape=(num_units), initializer=tf.zeros_initializer())
            keys = tf.matmul(keys, W2) + b2

        # Value Projection: (N, Tx, a)
        with tf.variable_scope("val_proj"):
            vals = tf.layers.dense(vals, num_units)

        # Get Attention weights
        attention_weights = tf.matmul(queries, keys, transpose_b=True)  # (N, Ty, Tx)
        _, Ty, Tx = attention_weights.get_shape().as_list()

        if training: # vanilla attention
            alignments = tf.nn.softmax(attention_weights)
            max_attentions = prev_max_attentions
        else: # force monotonic attention
            key_masks = tf.sequence_mask(prev_max_attentions, Tx)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, Ty, 1])
            paddings = tf.ones_like(attention_weights) * (-2 ** 32 + 1)  # (N, Ty, Tx)
            attention_weights = tf.where(tf.equal(key_masks, False), attention_weights, paddings)
            alignments = tf.nn.softmax(attention_weights)
            max_attentions = tf.argmax(alignments, -1)  # (N, Ty)

        # Get Context Vectors
        tensor = tf.layers.dropout(alignments, rate=dropout_rate, training=training)
        tensor = tf.matmul(tensor, vals)  # (N, Ty, a)
        tensor *= tf.sqrt(1. / tf.to_float(Tx))

        # Restore shape for residual connection
        tensor = fc_block(tensor,
                          num_units=hp.embed_size,
                          dropout_rate=0,
                          training=training,
                          scope="tensor_fc_block")  # (N, Tx, e)

        # returns the alignment of the first one
        alignment = tf.transpose(alignments[0])[::-1, :] # (Tx, Ty)
    return tensor, alignment, max_attentions