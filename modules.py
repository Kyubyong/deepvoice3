# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import math


def embed(inputs, vocab_size, num_units, zero_pad=False, scope="embedding", reuse=None):
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
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavxavier())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)   
 
def normalize(inputs, 
              type="bn",
              decay=.999,
              epsilon=1e-8,
              training=True, 
              activation_fn=None,
              reuse=None,
              scope="normalize"):
    '''Applies {batch|layer|weight} normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but 
        the last dimension. Or if type is `ln`, the normalization is over 
        the last dimension. Note that this is different from the native 
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py` 
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type=="bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        
        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank==2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank==3:
                inputs = tf.expand_dims(inputs, axis=1)
            
            outputs = tf.contrib.layers.batch_norm(inputs=inputs, 
                                               decay=decay,
                                               center=True, 
                                               scale=True, 
                                               updates_collections=None,
                                               training=training,
                                               scope=scope,
                                               fused=True,
                                               reuse=reuse)
            # restore original shape
            if inputs_rank==2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank==3:
                outputs = tf.squeeze(outputs, axis=1)
        else: # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs, 
                                               decay=decay,
                                               center=True, 
                                               scale=True, 
                                               updates_collections=None,
                                               training=training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)    
    elif type in ("ln",  "ins"):
        reduction_axis = -1 if type=="ln" else 1   
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [reduction_axis], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
    else:
        outputs = inputs
    
    if activation_fn:
        outputs = activation_fn(outputs)
                
    return outputs

def conv1d(inputs, 
           filters=None, 
           size=1, 
           rate=1, 
           padding="SAME", 
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower()=="causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"
        
        if filters is None:
            filters = inputs.get_shape().as_list[-1]
        
        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":activation_fn, 
                "use_bias":use_bias, "reuse":reuse, "kernel_initializer":tf.contrib.layers.xavier_initializer}
        
        outputs = tf.layers.conv1d(**params)
    return outputs

def conv_block(inputs,
               size=5,
               training=False,
               scope="conv_block"):
    _inputs = inputs
    with tf.variable_scope(scope):
        inputs = tf.layers.dropout(inputs, rate=hp.dropout_rate, training=training)
        inputs = conv1d(inputs, tf.shape(inputs)[-1]*2, size=size, scope="conv1d_{}".format(i))  # (N, T_x, 64*2)
        inputs = normalize(inputs, type=hp.norm_type, training=training, activation_fn=None)
        A, B = tf.split(inputs, 2, -1) # (N, T_x, 64) * 2
        inputs = A*tf.nn.sigmoid(B) # (N, T_x, 64) <- gated linear unit activation
        inputs += _inputs # residual connection
        inputs *= math.sqrt(0.5) # scale

    return inputs

def positional_encoding(inputs,
                        vocab_size,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''
    Positional_Encoding for a given tensor.
    Args:
      inputs: [Tensor], A tensor contains the ids to be search from the lookup table, shape = [batch_size, 1 + len(inpt)]
      vocab_size: [Int], Vocabulary size
      num_units: [Int], Hidden size of embedding
      zero_pad: [Boolean], If True, all the values of the first row(id = 0) should be constant zero
      scale: [Boolean], If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: [String], Optional scope for 'variable_scope'
      reuse: [Boolean], If to reuse the weights of a previous layer by the same name
      Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    with tf.variable_scope(scope, reuse=reuse):

        input_one = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / num_units) for i in range(num_units)]
            for pos in range(max_len)])
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, input_one)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs

def prenet(inputs, num_units=None, training=True, scope="prenet", reuse=None):
    '''Prenet for Encoder and Decoder1.
    Args:
      inputs: A 2D or 3D tensor.
      num_units: A list of two integers. or None.
      training: A python boolean.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3D tensor of shape [N, T, num_units/2].
    '''
    if num_units is None:
        num_units = [hp.embed_size, hp.embed_size//2]
        
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=training, name="dropout2") 
    return outputs # (N, ..., num_units[1])
