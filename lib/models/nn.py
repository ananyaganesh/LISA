#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
# Copyright 2016 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lib import linalg
from lib.etc.tarjan import Tarjan
from lib.models import rnn
from configurable import Configurable
from vocab import Vocab

import scipy.linalg
import scipy.sparse.csgraph

float32_eps = np.finfo(np.float32).eps


def layer_norm(inputs, reuse, epsilon=1e-6):
  """Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
  with tf.variable_scope("layer_norm", reuse=reuse):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
    normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
    outputs = gamma * normalized + beta
  return outputs


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      np.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return x + signal


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  experessed in terms of b, sin(a) and cos(a).
  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image
  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  static_shape = x.get_shape().as_list()
  num_dims = len(static_shape) - 2
  channels = tf.shape(x)[-1]
  num_timescales = channels // (num_dims * 2)
  log_timescale_increment = (
      np.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  for dim in xrange(num_dims):
    length = tf.shape(x)[dim + 1]
    position = tf.to_float(tf.range(length))
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    prepad = dim * 2 * num_timescales
    postpad = channels - (dim + 1) * 2 * num_timescales
    signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
    for _ in xrange(1 + dim):
      signal = tf.expand_dims(signal, 0)
    for _ in xrange(num_dims - 1 - dim):
      signal = tf.expand_dims(signal, -2)
    x += signal
  return x


def attention_bias_ignore_padding(lengths):
  """Create an bias tensor to be added to attention logits.
  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].
  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
  memory_padding = tf.cast(tf.logical_not(mask), tf.float32)
  ret = memory_padding * -1e9
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.
  Args:
    x: a Tensor with shape [..., a, b]
  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def split_heads_2d(x, num_heads):
  """Split channels (dimension 4) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, height, width, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, height, width, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def combine_heads_2d(x):
  """Inverse of split_heads_2d.
  Args:
    x: a Tensor with shape
      [batch, num_heads, height, width, channels / num_heads]
  Returns:
    a Tensor with shape [batch, height, width, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def combine_heads(x):
  """Inverse of split_heads.
  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def dot_product_attention(q, k, v,
                          bias,
                          dropout_rate=1.0,
                          num_capsule_heads=0,
                          manual_attn=None,
                          hard_attn=False,
                          name=None):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    # if add_attn is not None:
    #   # heads x batch x seq_len x seq_len
    #   weights_transpose = tf.transpose(logits, [1, 0, 2, 3])
    #   weights_rest = weights_transpose[1:]
    #   weights_comb = tf.concat([tf.expand_dims(add_attn + weights_transpose[0], 0), weights_rest], axis=0)
    #   logits = tf.transpose(weights_comb, [1, 0, 2, 3])

    if bias is not None:
      logits += bias
    # first num_capsule_heads capsuled, rest regular
    weights1 = tf.nn.softmax(logits[:, :num_capsule_heads, :, :], dim=2)
    weights2 = tf.nn.softmax(logits[:, num_capsule_heads:, :, :], dim=3)
    weights = tf.concat([weights1, weights2], axis=1, name="attention_weights")
    # weights is batch x heads x seq_len x seq_len
    if manual_attn is not None:
      # heads x batch x seq_len x seq_len
      weights_transpose = tf.transpose(weights, [1, 0, 2, 3])
      weights_rest = weights_transpose[1:]
      weights_comb = tf.concat([tf.expand_dims(manual_attn, 0), weights_rest], axis=0)
      weights = tf.transpose(weights_comb, [1, 0, 2, 3])
    if hard_attn:
      # heads x batch x seq_len x seq_len
      weights_transpose = tf.transpose(weights, [1, 0, 2, 3])
      weights_rest = weights_transpose[1:]
      w = weights_transpose[0]
      hard_weights = tf.where(tf.equal(w, tf.tile(tf.expand_dims(tf.reduce_max(w, axis=-1), -1), [1, 1, tf.shape(w)[-1]])), tf.ones_like(w), tf.zeros_like(w))
      # hard_weights = tf.Print(hard_weights, [hard_weights], "hard weights", summarize=100)
      # hard_weights = tf.Print(hard_weights, [w], "w", summarize=100)
      weights_comb = tf.concat([tf.expand_dims(hard_weights, 0), weights_rest], axis=0)
      weights = tf.transpose(weights_comb, [1, 0, 2, 3])
    # dropping out the attention links for each of the heads
    weights_drop = tf.nn.dropout(weights, dropout_rate)
    return tf.matmul(weights_drop, v), logits


def compute_qkv(antecedent, total_key_depth, total_value_depth):
  """Computes query, key and value.
  Args:
    total_key_depth: an integer
    total_value_depth: and integer
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  params = tf.get_variable("qkv_transform", [1, 1, total_key_depth, 2*total_key_depth + total_value_depth])
  antecedent = tf.expand_dims(antecedent, 1)
  qkv_combined = tf.nn.conv2d(antecedent, params, [1, 1, 1, 1], "SAME")
  qkv_combined = tf.squeeze(qkv_combined, 1)
  q, k, v = tf.split(qkv_combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
  return q, k, v


def compute_qkv_2d(antecedent, total_key_depth, total_value_depth):
  """Computes query, key and value.
  Args:
    query_antecedent: a Tensor with shape [batch, h, w, depth_k]
    memory_antecedent: a Tensor with shape [batch, h, w, depth_k]
    total_key_depth: an integer
    total_value_depth: and integer
  Returns:
    q, k, v : [batch, h, w, depth_k] tensors
  """
  params = tf.get_variable("qkv_transform", [1, 1, total_key_depth, 2*total_key_depth + total_value_depth])
  qkv_combined = tf.nn.conv2d(antecedent, params, [1, 1, 1, 1], "SAME")
  q, k, v = tf.split(qkv_combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
  return q, k, v


def multihead_attention_2d(antecedent,
                           bias,
                           total_key_depth,
                           total_value_depth,
                           output_depth,
                           num_heads,
                           dropout_rate,
                           name=None):
  """2d Multihead scaled-dot-product attention with inp/output transformations.
  Args:
    query_antecedent: a Tensor with shape [batch, h, w, depth_k]
    memory_antecedent: a Tensor with shape [batch, h, w, depth_k]
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    attention_type: String, type of attention function to use.
    query_shape: an tuple indicating the height and width of each query block.
    memory_flange: an integer indicating how much to look in height and width
    name: an optional string
  Returns:
    A Tensor of shape [batch, h, w, depth_k]
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  with tf.variable_scope(
      name,
      default_name="multihead_attention_2d",
      values=[antecedent]):
    q, k, v = compute_qkv_2d(antecedent, total_key_depth, total_value_depth)
    # after splitting, shape is [batch, heads, h, w, depth]
    q = split_heads_2d(q, num_heads)
    k = split_heads_2d(k, num_heads)
    v = split_heads_2d(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    # flatten out h/w dim
    shape = tf.shape(q)
    q = tf.reshape(q, [shape[0], shape[1], -1, shape[-1]])
    k = tf.reshape(k, [shape[0], shape[1], -1, shape[-1]])
    v = tf.reshape(v, [shape[0], shape[1], -1, shape[-1]])
    x = dot_product_attention(q, k, v, bias, dropout_rate)
    # unflatten
    x = tf.reshape(x, [shape[0], shape[1], shape[2], shape[3], -1])
    x = combine_heads_2d(x)
    x = tf.layers.conv2d(
        x,
        output_depth,
        (1, 1),
        name="output_transform")
    return x


def multihead_attention(antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        num_capsule_heads,
                        manual_attn=None,
                        hard_attn=False,
                        name=None):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(name, default_name="multihead_attention", values=[antecedent]):
    q, k, v = compute_qkv(antecedent, total_key_depth, total_value_depth)
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    x, attn_weights = dot_product_attention(q, k, v, bias, dropout_rate, num_capsule_heads, manual_attn, hard_attn)
    x = combine_heads(x)
    params = tf.get_variable("final_proj", [1, 1, total_key_depth, output_depth])
    x = tf.expand_dims(x, 1)
    x = tf.nn.conv2d(x, params, [1, 1, 1, 1], "SAME")
    x = tf.squeeze(x, 1)
    return x, attn_weights


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     dropout,
                     nonlinearity,
                     kernel):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope("conv_hidden_relu", [inputs]):
    inputs = tf.expand_dims(inputs, 1)
    in_size = inputs.get_shape().as_list()[-1]
    params1 = tf.get_variable("ff1", [1, 1, in_size, hidden_size])
    params2 = tf.get_variable("ff2", [1, kernel, hidden_size, hidden_size])
    params3 = tf.get_variable("ff3", [1, 1, hidden_size, output_size])
    h = tf.nn.conv2d(inputs, params1, [1, 1, 1, 1], "SAME")
    h = nonlinearity(h)
    h = tf.nn.dropout(h, dropout)
    h = tf.nn.conv2d(h, params2, [1, 1, 1, 1], "SAME")
    h = nonlinearity(h)
    h = tf.nn.dropout(h, dropout)
    ret = tf.nn.conv2d(h, params3, [1, 1, 1, 1], "SAME")
    ret = tf.squeeze(ret, 1)
    return ret


#***************************************************************
class NN(Configurable):
  """"""
  
  ZERO = tf.convert_to_tensor(0.)
  ONE = tf.convert_to_tensor(1.)
  PUNCT = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    # global_step = kwargs.pop('global_step', None)
    super(NN, self).__init__(*args, **kwargs)
    
    # if global_step is not None:
    #   self._global_sigmoid = 1-tf.nn.sigmoid(3*(2*global_step/(self.train_iters-1)-1))
    # else:
    #   self._global_sigmoid = 1
    
    self.tokens_to_keep3D = None
    self.sequence_lengths = None
    self.n_tokens = None
    self.moving_params = None
    return
  
  #=============================================================
  def embed_concat(self, word_inputs, tag_inputs=None, rel_inputs=None):
    """"""
    
    if self.moving_params is None:
      word_keep_prob = self.word_keep_prob
      tag_keep_prob = self.tag_keep_prob
      rel_keep_prob = self.rel_keep_prob
      noise_shape = tf.stack([tf.shape(word_inputs)[0], tf.shape(word_inputs)[1], 1])
      
      if word_keep_prob < 1:
        word_mask = tf.nn.dropout(tf.ones(noise_shape), word_keep_prob)*word_keep_prob
      else:
        word_mask = 1
      if tag_inputs is not None and tag_keep_prob < 1:
        tag_mask = tf.nn.dropout(tf.ones(noise_shape), tag_keep_prob)*tag_keep_prob
      else:
        tag_mask = 1
      if rel_inputs is not None and rel_keep_prob < 1:
        rel_mask = tf.nn.dropout(tf.ones(noise_shape), rel_keep_prob)*rel_keep_prob
      else:
        rel_mask = 1
      
      word_embed_size = word_inputs.get_shape().as_list()[-1]
      tag_embed_size = 0 if tag_inputs is None else tag_inputs.get_shape().as_list()[-1]
      rel_embed_size = 0 if rel_inputs is None else rel_inputs.get_shape().as_list()[-1]
      total_size = word_embed_size + tag_embed_size + rel_embed_size
      if word_embed_size == tag_embed_size:
        total_size += word_embed_size
      dropped_sizes = word_mask * word_embed_size + tag_mask * tag_embed_size + rel_mask * rel_embed_size
      if word_embed_size == tag_embed_size:
        dropped_sizes += word_mask * tag_mask * word_embed_size
      scale_factor = total_size / (dropped_sizes + self.epsilon)
      
      word_inputs *= word_mask * scale_factor
      if tag_inputs is not None:
        tag_inputs *= tag_mask * scale_factor
      if rel_inputs is not None:
        rel_inputs *= rel_mask * scale_factor
    else:
      word_embed_size = word_inputs.get_shape().as_list()[-1]
      tag_embed_size = 0 if tag_inputs is None else tag_inputs.get_shape().as_list()[-1]
      rel_embed_size = 0 if rel_inputs is None else rel_inputs.get_shape().as_list()[-1]
    
    return tf.concat(axis=2, values=filter(lambda x: x is not None, [word_inputs, tag_inputs, rel_inputs]))
  
  #=============================================================
  def RNN(self, inputs):
    """"""
    
    input_size = inputs.get_shape().as_list()[-1]
    cell = self.recur_cell(self._config, input_size=input_size, moving_params=self.moving_params)
    lengths = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])
    
    if self.moving_params is None:
      ff_keep_prob = self.ff_keep_prob
      recur_keep_prob = self.recur_keep_prob
    else:
      ff_keep_prob = 1
      recur_keep_prob = 1
    
    if self.recur_bidir:
      top_recur, fw_recur, bw_recur = rnn.dynamic_bidirectional_rnn(cell, cell, inputs,
                                                                    lengths,
                                                                    ff_keep_prob=ff_keep_prob,
                                                                    recur_keep_prob=recur_keep_prob,
                                                                    dtype=tf.float32)
      fw_cell, fw_out = tf.split(axis=1, num_or_size_splits=2, value=fw_recur)
      bw_cell, bw_out = tf.split(axis=1, num_or_size_splits=2, value=bw_recur)
      end_recur = tf.concat(axis=1, values=[fw_out, bw_out])
      top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(2*self.recur_size)])
    else:
      top_recur, end_recur = rnn.dynamic_rnn(cell, inputs,
                                             lengths,
                                             ff_keep_prob=ff_keep_prob,
                                             recur_keep_prob=recur_keep_prob,
                                             dtype=tf.float32)
      top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(self.recur_size)])
    return top_recur, end_recur

  # =============================================================
  def CNN(self, inputs, kernel1, kernel2, output_size, dropout_keep_rate, nonlinearity):
    """"""
    input_size = inputs.get_shape().as_list()[-1]

    initializer = tf.contrib.layers.xavier_initializer()
    # mat = linalg.orthonormal_initializer(input_size, output_size)
    # initializer = tf.constant_initializer(mat)

    if self.moving_params is not None:
      dropout_keep_rate = 1.0

    params = tf.get_variable('CNN', [kernel1, kernel2, input_size, output_size], initializer=initializer)
    if kernel1 == 1:
      inputs = tf.expand_dims(inputs, 1)
    conv_out = tf.nn.conv2d(inputs, params, [1, 1, 1, 1], 'SAME')
    if kernel1 == 1:
      conv_out = tf.squeeze(conv_out, 1)
    conv_out = nonlinearity(conv_out)
    conv_out = tf.nn.dropout(conv_out, dropout_keep_rate)
    return conv_out

  # =============================================================
  def transformer(self, inputs, hidden_size, num_heads, attn_dropout, relu_dropout, prepost_dropout, relu_hidden_size,
                  nonlinearity, kernel, reuse, num_capsule_heads, manual_attn=None, hard_attn=False):
    """"""
    # input_size = inputs.get_shape().as_list()[-1]
    lengths = tf.reshape(tf.to_int64(self.sequence_lengths), [-1])
    mask = attention_bias_ignore_padding(lengths)

    # mat = linalg.orthonormal_initializer(input_size, output_size)
    # initializer = tf.constant_initializer(mat)
    if self.moving_params is not None:
      attn_dropout = 1.0
      relu_dropout = 1.0
      prepost_dropout = 1.0

    with tf.variable_scope("self_attention"):
      x = layer_norm(inputs, reuse)
      y, attn_weights = multihead_attention(x, mask, hidden_size, hidden_size, hidden_size, num_heads, attn_dropout, num_capsule_heads, manual_attn, hard_attn)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))

    with tf.variable_scope("ffnn"):
      x = layer_norm(x, reuse)
      y = conv_hidden_relu(x, relu_hidden_size, hidden_size, relu_dropout, nonlinearity, kernel)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))

    return x, attn_weights
  
  #=============================================================
  def soft_attn(self, top_recur):
    """"""
    
    reuse = (self.moving_params is not None) or None
    
    input_size = top_recur.get_shape().as_list()[-1]
    with tf.variable_scope('MLP', reuse=reuse):
      head_mlp, dep_mlp = self.MLP(top_recur, self.info_mlp_size,
                                   func=self.info_func,
                                   keep_prob=self.info_keep_prob,
                                   n_splits=2)
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(dep_mlp, head_mlp, keep_prob=self.info_keep_prob)
      arc_prob = self.softmax(arc_logits)
      head_lin = tf.matmul(arc_prob, top_recur)
      top_recur = tf.concat(axis=2, values=[top_recur, head_lin])
    top_recur.set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(4*self.recur_size)])
    return top_recur

  #=============================================================
  def linear(self, inputs, output_size, n_splits=1, add_bias=False):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_shape = tf.stack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    shape_to_set = [tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)]
    
    if self.moving_params is None:
      keep_prob = self.info_keep_prob
    else:
      keep_prob = 1
    
    if keep_prob < 1:
      noise_shape = tf.stack([batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)

    lin = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits,
                        add_bias=add_bias,
                        moving_params=self.moving_params)
    if n_splits == 1:
      lin = [lin]
    for i, split in enumerate(lin):
      split.set_shape(shape_to_set)
    if n_splits == 1:
      return lin[0]
    else:
      return lin

  #=============================================================
  def softmax(self, inputs):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = input_shape[2]
    inputs = tf.reshape(inputs, tf.stack([-1, input_size]))
    probs = tf.nn.softmax(inputs)
    probs = tf.reshape(probs, tf.stack([batch_size, bucket_size, input_size]))
    return probs
  
  #=============================================================
  def MLP(self, inputs, output_size, func=None, keep_prob=None, n_splits=1):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_shape = tf.stack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    shape_to_set = [tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)]
    if func is None:
      func = self.mlp_func
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if keep_prob < 1:
      noise_shape = tf.stack([batch_size] + [1]*(n_dims-2) + [input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    linear = linalg.linear(inputs,
                        output_size,
                        n_splits=n_splits * (1+(func.__name__ in ('gated_tanh', 'gated_identity'))),
                        add_bias=True,
                        moving_params=self.moving_params)
    if func.__name__ in ('gated_tanh', 'gated_identity'):
      linear = [tf.concat(axis=n_dims-1, values=[lin1, lin2]) for lin1, lin2 in zip(linear[:len(linear)//2], linear[len(linear)//2:])]
    if n_splits == 1:
      linear = [linear]
    for i, split in enumerate(linear):
      split = func(split)
      split.set_shape(shape_to_set)
      linear[i] = split
    if n_splits == 1:
      return linear[0]
    else:
      return linear
  
  #=============================================================
  def double_MLP(self, inputs, n_splits=1):
    """"""
    
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = self.attn_mlp_size
    output_shape = tf.stack([batch_size, bucket_size, bucket_size, output_size])
    shape_to_set = [tf.Dimension(None), tf.Dimension(None), tf.Dimension(None), tf.Dimension(output_size)]
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    lin1, lin2 = linalg.linear(inputs,
                               output_size*n_splits,
                               n_splits=2,
                               add_bias=True,
                               moving_params=self.moving_params)
    lin1 = tf.reshape(tf.transpose(lin1, [0, 2, 1]), tf.stack([-1, bucket_size, 1]))
    lin2 = tf.reshape(tf.transpose(lin2, [0, 2, 1]), tf.stack([-1, 1, bucket_size]))
    lin = lin1 + lin2
    lin = tf.reshape(lin, tf.stack([batch_size, n_splits*output_size, bucket_size, bucket_size]))
    lin = tf.transpose(lin, [0,2,3,1])
    top_mlps = tf.split(axis=3, num_or_size_splits=n_splits, value=self.mlp_func(lin))
    for top_mlp in top_mlps:
      top_mlp.set_shape(shape_to_set)
    if n_splits == 1:
      return top_mlps[0]
    else:
      return top_mlps
  
  #=============================================================
  def linear_classifier(self, inputs, n_classes, add_bias=True, keep_prob=None):
    """"""
    
    n_dims = len(inputs.get_shape().as_list())
    batch_size = tf.shape(inputs)[0]
    bucket_size = tf.shape(inputs)[1]
    input_size = inputs.get_shape().as_list()[-1]
    output_size = n_classes
    output_shape = tf.stack([batch_size] + [bucket_size]*(n_dims-2) + [output_size])
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size] + [1]*(n_dims-2) +[input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    inputs = tf.reshape(inputs, [-1, input_size])
    output = linalg.linear(inputs,
                    output_size,
                    add_bias=add_bias,
                    initializer=tf.zeros_initializer(),
                    moving_params=self.moving_params)
    output = tf.reshape(output, output_shape)
    output.set_shape([tf.Dimension(None)]*(n_dims-1) + [tf.Dimension(output_size)])
    return output
  
  #=============================================================
  def bilinear_classifier(self, inputs1, inputs2, add_bias1=True, add_bias2=False, keep_prob=None):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    
    if self.moving_params is None:
      if keep_prob is None:
        keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      # Experimental
      #inputs1 = tf.nn.dropout(inputs1, keep_prob if add_bias2 else tf.sqrt(keep_prob), noise_shape=noise_shape)
      #inputs2 = tf.nn.dropout(inputs2, keep_prob if add_bias1 else tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)
    
    bilin = linalg.bilinear(inputs1, inputs2, 1,
                            add_bias1=add_bias1,
                            add_bias2=add_bias2,
                            initializer=tf.zeros_initializer(),
                            moving_params=self.moving_params)
    output = tf.squeeze(bilin)
    return output
  
  #=============================================================
  def diagonal_bilinear_classifier(self, inputs1, inputs2, add_bias1=True, add_bias2=False):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    shape_to_set = tf.stack([batch_size, bucket_size, input_size+1])
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, tf.sqrt(keep_prob), noise_shape=noise_shape)
    
    bilin = linalg.diagonal_bilinear(inputs1, inputs2, 1,
                                     add_bias1=add_bias1,
                                     add_bias2=add_bias2,
                                     initializer=tf.zeros_initializer(),
                                     moving_params=self.moving_params)
    output = tf.squeeze(bilin)
    return output
  
  #=============================================================
  def conditional_linear_classifier(self, inputs, n_classes, probs, add_bias=True):
    """"""
    
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs.get_shape().as_list()[-1]
    
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, 1, input_size])
      inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    
    lin = linalg.linear(inputs,
                        n_classes,
                        add_bias=add_bias,
                        initializer=tf.zeros_initializer(),
                        moving_params=self.moving_params)
    weighted_lin = tf.matmul(lin, tf.expand_dims(probs, 3), adjoint_a=True)
    
    return weighted_lin, lin
  
  #=============================================================
  def conditional_diagonal_bilinear_classifier(self, inputs1, inputs2, n_classes, probs, add_bias1=True, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size+1]
    output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, tf.sqrt(keep_prob), noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, tf.sqrt(keep_prob), noise_shape=noise_shape)
    
    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs1.set_shape(input_shape_to_set)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs2.set_shape(input_shape_to_set)
    
    bilin = linalg.diagonal_bilinear(inputs1, inputs2,
                                     n_classes,
                                     add_bias1=add_bias1,
                                     add_bias2=add_bias2,
                                     initializer=tf.zeros_initializer(),
                                     moving_params=self.moving_params)
    weighted_bilin = tf.matmul(bilin, tf.expand_dims(probs, 3))
    
    return weighted_bilin, bilin
  
  #=============================================================
  def conditional_bilinear_classifier(self, inputs1, inputs2, n_classes, probs, add_bias1=True, add_bias2=True):
    """"""
    
    input_shape = tf.shape(inputs1)
    batch_size = input_shape[0]
    bucket_size = input_shape[1]
    input_size = inputs1.get_shape().as_list()[-1]
    input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size+1]
    output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
    if len(probs.get_shape().as_list()) == 2:
      probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    else:
      probs = tf.stop_gradient(probs)
    
    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape = tf.stack([batch_size, 1, input_size])
      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)
    
    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs1.set_shape(input_shape_to_set)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
    inputs2.set_shape(input_shape_to_set)
    
    bilin = linalg.bilinear(inputs1, inputs2,
                     n_classes,
                     add_bias1=add_bias1,
                     add_bias2=add_bias2,
                     initializer=tf.zeros_initializer(),
                     moving_params=self.moving_params)
    weighted_bilin = tf.matmul(bilin, tf.expand_dims(probs, 3))
    
    return weighted_bilin, bilin

  # =============================================================
  def bilinear_classifier_nary(self, inputs1, inputs2, n_classes, combine=False, combined_weights=None, add_bias1=True, add_bias2=True):
    """"""

    #inputs1 = tf.Print(inputs1, [tf.shape(inputs1)], "Pred reps shape")
    #inputs2 = tf.Print(inputs2, [tf.shape(inputs2)], "Role reps shape")
    input_shape1 = tf.shape(inputs1)
    input_shape2 = tf.shape(inputs2)

    batch_size1 = input_shape1[0]
    batch_size2 = input_shape2[0]


    # with tf.control_dependencies([tf.assert_equal(input_shape1[1], input_shape2[1])]):
    bucket_size1 = input_shape1[1]
    bucket_size2 = input_shape2[1]
    input_size1 = inputs1.get_shape().as_list()[-1]
    input_size2 = inputs2.get_shape().as_list()[-1]

    input_shape_to_set1 = [tf.Dimension(None), tf.Dimension(None), input_size1 + 1]
    input_shape_to_set2 = [tf.Dimension(None), tf.Dimension(None), input_size2 + 1]

    # output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
    # if len(probs.get_shape().as_list()) == 2:
    #   probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
    # else:
    #   probs = tf.stop_gradient(probs)

    if self.moving_params is None:
      keep_prob = self.mlp_keep_prob
    else:
      keep_prob = 1
    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
      noise_shape1 = tf.stack([batch_size1, 1, input_size1])
      noise_shape2 = tf.stack([batch_size2, 1, input_size2])

      inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape1)
      inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape2)

    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, bucket_size1, 1]))])
    inputs1.set_shape(input_shape_to_set1)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, bucket_size2, 1]))])
    inputs2.set_shape(input_shape_to_set2)

    bilin = linalg.bilinear_noreshape(inputs1, inputs2,
                            n_classes,
                            combine=combine,
                            combined_weights=combined_weights,
                            add_bias1=add_bias1,
                            add_bias2=add_bias2,
                            initializer=tf.zeros_initializer(),
                            moving_params=self.moving_params)
    # weighted_bilin = tf.matmul(bilin, tf.expand_dims(probs, 3))

    return bilin
  
  #=============================================================
  def output(self, logits3D, targets3D, num_classes=1):
    """"""

    targets_shape = tf.shape(targets3D)
    batch_size = targets_shape[0]
    bucket_size = targets_shape[1]
    original_shape = tf.cond(tf.greater(tf.rank(logits3D), 1), lambda: tf.shape(logits3D), lambda: tf.stack([batch_size, bucket_size, bucket_size, num_classes]))
    flat_shape = tf.stack([batch_size, bucket_size])

    tokens_to_keep1D = tf.reshape(self.tokens_to_keep3D, [-1])

    def dummy_loss():
      # loss, accuracy, tf.reshape(predictions1D, flat_shape), tf.reshape(probabilities2D, original_shape), correct1D, n_correct
      return tf.constant(0.), tf.constant(0.), tf.zeros(flat_shape, dtype=tf.int32), tf.zeros(original_shape), \
             tf.zeros_like(tokens_to_keep1D, dtype=tf.float32), tf.constant(0.)

    def compute_loss():

      logits2D = tf.reshape(logits3D, tf.stack([batch_size*bucket_size, -1]))
      targets1D = tf.reshape(targets3D, [-1])

      predictions1D = tf.to_int32(tf.argmax(logits2D, 1))
      probabilities2D = tf.nn.softmax(logits2D)
      cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2D, labels=targets1D)

      correct1D = tf.to_float(tf.equal(predictions1D, targets1D))
      n_correct = tf.reduce_sum(correct1D * tokens_to_keep1D)
      accuracy = n_correct / self.n_tokens
      loss = tf.reduce_sum(cross_entropy1D * tokens_to_keep1D) / self.n_tokens
      return loss, accuracy, tf.reshape(predictions1D, flat_shape), tf.reshape(probabilities2D, original_shape), correct1D, n_correct

    loss, accuracy, predictions, probabilities, correct, n_correct = tf.cond(tf.greater(tf.rank(logits3D), 1),
      lambda: compute_loss(),
      lambda: dummy_loss())

    output = {
      'probabilities': probabilities,
      'predictions': predictions,
      'tokens': tokens_to_keep1D,
      'correct': correct * tokens_to_keep1D,
      'n_correct': n_correct,
      'n_tokens': self.n_tokens,
      'accuracy': accuracy,
      'count': self.n_tokens,
      'loss': loss
    }

    return output

  # =============================================================
  def output_transpose(self, logits3D, targets3D):
    """"""

    original_shape = tf.shape(logits3D)
    batch_size = original_shape[0]
    bucket_size = original_shape[1]
    flat_shape = tf.stack([batch_size, bucket_size])

    logits3D = tf.transpose(logits3D, [0, 2, 1])

    logits2D = tf.reshape(logits3D, tf.stack([batch_size * bucket_size, -1]))
    targets1D = tf.reshape(targets3D, [-1])
    tokens_to_keep1D = tf.reshape(self.tokens_to_keep3D, [-1])

    predictions1D = tf.to_int32(tf.argmax(logits2D, 1))
    probabilities2D = tf.nn.softmax(logits2D)
    cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2D, labels=targets1D)

    correct1D = tf.to_float(tf.equal(predictions1D, targets1D))
    n_correct = tf.reduce_sum(correct1D * tokens_to_keep1D)
    accuracy = n_correct / self.n_tokens
    loss = tf.reduce_sum(cross_entropy1D * tokens_to_keep1D) / self.n_tokens

    output = {
      'probabilities': tf.reshape(probabilities2D, original_shape),
      'predictions': tf.reshape(predictions1D, flat_shape),
      'tokens': tokens_to_keep1D,
      'correct': correct1D * tokens_to_keep1D,
      'n_correct': n_correct,
      'n_tokens': self.n_tokens,
      'accuracy': accuracy,
      'loss': loss
    }

    return output

  #=============================================================
  def output_srl(self, logits, targets, trigger_label_indices, outside_label_idx, transition_params):
    """"""

    # logits are batch x seq_len x num_classes x seq_len
    # targets are batch x seq_len x num_targets

    # transpose to batch x seq_len x seq_len x num_classes
    logits_transposed = tf.transpose(logits, [0, 1, 3, 2])

    original_shape = tf.shape(logits)
    batch_size = original_shape[0]
    bucket_size = original_shape[1]
    # num_classes = original_shape[2]
    # flat_shape = tf.stack([batch_size, bucket_size])
    # original_shape = logits.get_shape().as_list()[2]
    # batch_size = original_shape[0]
    # bucket_size = original_shape[1]
    num_classes = logits.get_shape().as_list()[2]

    # flatten logits along last dimension: batch x seq_len x seq_len*num_classes
    # logits_flattened = tf.reshape(logits_transposed, [batch_size, bucket_size, -1])

    # need to turn targets into this 2d representation.
    # have: labels for each token for each trigger (<= sentence len) and trigger_label_idx
    # need: batch_size x seq_len x seq_len labels: the actual labels for each trigger, and all O otherwise
    # todo don't hardcode 7

    # now we have k sets of targets for the k frames
    # (t1) f1 f2 f3
    # (t2) f1 f2 f3
    srl_targets = targets[:,:,3:]

    # get indices of trigger labels in srl_targets
    tile_multiples = tf.concat([tf.ones(tf.shape(tf.shape(srl_targets)), dtype=tf.int32), tf.shape(trigger_label_indices)], axis=0)
    targets_tile = tf.tile(tf.expand_dims(srl_targets, -1), tile_multiples)
    trigger_indices = tf.cast(tf.where(tf.reduce_any(tf.equal(targets_tile, trigger_label_indices), -1)), tf.int32)
    # srl_targets_tile = tf.tile(srl_targets, [1, 1, 1, num_trigger_labels])
    # trigger_indices = tf.cast(tf.where(tf.reduce_any(tf.equal(srl_targets_tile, trigger_label_indices), -1)), tf.int32)
    # trigger_indices = tf.cast(tf.where(tf.equal(srl_targets, trigger_label_idx)), tf.int32)

    # trigger_indices = tf.Print(trigger_indices, [trigger_indices], "trigger_indices", summarize=5000)

    # get all the tags for each token (which is the trigger for a frame), structuring
    # targets3D as follows (assuming t1 and t2 are triggers for f1 and f3, repsectively):
    # (t1) f1 f1 f1
    # (t2) f3 f3 f3
    actual_targets = tf.gather_nd(tf.transpose(srl_targets, [0, 2, 1]), tf.stack([trigger_indices[:,0],trigger_indices[:,2]], -1))
    i1 = tf.tile(tf.expand_dims(trigger_indices[:,0], -1), [1, bucket_size])
    i2 = tf.tile(tf.expand_dims(trigger_indices[:,1], -1), [1, bucket_size])
    i3 = tf.tile(tf.expand_dims(tf.range(bucket_size), 0), [tf.shape(trigger_indices)[0], 1])
    trigger_idx = tf.stack([i1, i2, i3], axis=-1)
    targets3D = tf.scatter_nd(trigger_idx, actual_targets, [batch_size, bucket_size, bucket_size])

    # batch x seq x seq: 0 where target, 1 otherwise
    not_targets3D = 1-tf.reduce_max(tf.scatter_nd(trigger_idx, tf.ones_like(actual_targets), [batch_size, bucket_size, bucket_size]), axis=-1)
    not_trigger_indices = tf.cast(tf.where(tf.equal(not_targets3D, 1)), tf.int32)
    num_not_triggers = tf.shape(not_trigger_indices)[0]

    # create a mask
    i1 = tf.tile(tf.expand_dims(not_trigger_indices[:, 0], -1), [1, bucket_size])
    i2 = tf.tile(tf.expand_dims(not_trigger_indices[:, 1], -1), [1, bucket_size])
    i3 = tf.tile(tf.expand_dims(tf.range(bucket_size), 0), [num_not_triggers, 1])
    not_trigger_idx = tf.stack([i1, i2, i3], axis=-1)

    # these are the ones we are going to MASK
    # subsample_trigger_rate = 0.0 -> mask nothing; subsample_trigger_rate = 1.0 -> mask everything
    num_to_sample = tf.cast(self.subsample_trigger_rate * tf.cast(num_not_triggers, tf.float32), tf.int32)
    sampled_indices = tf.random_shuffle(not_trigger_idx)[:num_to_sample]

    om = 1.0 - tf.scatter_nd(sampled_indices, tf.fill([num_to_sample, bucket_size], 1.0), [batch_size, bucket_size, bucket_size])

    targ_empty_indices = tf.cast(tf.where(tf.equal(targets3D, 0)), tf.int32)
    targets_mask3D = tf.scatter_nd(targ_empty_indices, tf.fill([tf.shape(targ_empty_indices)[0]], outside_label_idx), shape=tf.stack([batch_size, bucket_size, bucket_size]))
    targets3D_masked = targets3D + targets_mask3D

    overall_mask = om * self.tokens_to_keep3D * tf.transpose(self.tokens_to_keep3D, [0, 2, 1])
    non_masked_indices = tf.where(tf.not_equal(targets3D_masked * tf.cast(overall_mask, tf.int32), 0))
    non_masked_targets = tf.gather_nd(targets3D, non_masked_indices)
    count = tf.cast(tf.count_nonzero(non_masked_targets), tf.float32)

    # targets3D_masked = tf.Print(targets3D_masked, [targets3D_masked], "cross_entropy", summarize=5000)

    if transition_params is not None:
      # need to flatten batch x seq_len x seq_len x logits to
      # batch*seq_len x seq_len x logits,
      flattened_scores = tf.reshape(logits_transposed, tf.stack([batch_size*bucket_size, bucket_size, num_classes]))
      flattened_labels = tf.reshape(targets3D_masked, tf.stack([batch_size*bucket_size, bucket_size]))

      # and also get flattened sequence lengths.
      # this is batch x seq_len, need to tile rach seq_len seq_len times
      seq_lens = tf.reduce_sum(self.tokens_to_keep3D, 1)
      flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size*bucket_size]))
      log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(flattened_scores, flattened_labels,
                                                                            flat_seq_lens,
                                                                            transition_params=transition_params)
      loss = tf.reduce_mean(-log_likelihood)
    else:
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_transposed, labels=targets3D_masked)
      cross_entropy *= overall_mask
      # cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cross_entropy", summarize=5000)
      loss = tf.cond(tf.equal(count, 0.), lambda: tf.constant(0.), lambda: tf.reduce_sum(cross_entropy) / count)

    # training with predictions = batch x seq_len x seq_len
    # where each row is set of srl tags for that trigger
    # transposed, this gives us columns for each trigger; need to select out the columns
    # which contain the trigger label
    predictions = tf.cast(tf.argmax(logits_transposed, axis=-1), tf.int32)

    # predictions = tf.Print(predictions, [predictions], "predictions", summarize=5000)

    probabilities = tf.nn.softmax(logits_transposed)
    # gold_trigger_predictions

    correct = tf.reduce_sum(tf.cast(tf.equal(tf.gather_nd(predictions, non_masked_indices), non_masked_targets), tf.float32))

    # count  = tf.Print(count, [targets3D_masked], "targets3D_masked", summarize=4000)
    #
    # count  = tf.Print(count, [overall_mask], "overall_mask", summarize=4000)
    #
    # count  = tf.Print(count, [cross_entropy], "cross entropy", summarize=4000)
    # count = tf.Print(count, [count, correct, tf.reduce_sum(cross_entropy)])


    output = {
      'loss': loss,
      'probabilities': probabilities,
      'predictions': tf.transpose(predictions, [0, 2, 1]),
      'logits': tf.transpose(logits_transposed, [0, 2, 1, 3]),
      'transition_params': transition_params,
      # 'gold_trigger_predictions': tf.transpose(predictions, [0, 2, 1]),
      'count': count,
      'correct': correct
    }

    return output

  # =============================================================
  def output_srl_gather(self, logits_transposed, targets, trigger_predictions, transition_params, annotated3D=None, ignore_label_idx=None):
    """"""

    # logits are triggers_in_batch x num_classes x seq_len
    # targets are batch x seq_len x num_targets
    # trigger_predictions are batch x seq_len (1/0)

    # transpose to triggers_in_batch x seq_len x num_classes
    # logits_transposed = tf.transpose(logits, [0, 2, 1])

    original_shape = tf.shape(targets)
    batch_size = original_shape[0]
    bucket_size = original_shape[1]
    num_labels = tf.shape(logits_transposed)[-1]

    if annotated3D is not None:
      #tokens_to_keep = tf.Print(self.tokens_to_keep3D, [tf.count_nonzero(self.tokens_to_keep3D, dtype=tf.float32)], "tokenstokeep")
      #annotated = tf.Print(annotated3D, [tf.count_nonzero(annotated3D, dtype=tf.float32)], "annotated")
      tokens_to_keep3D = tf.cast(tf.logical_and(tf.cast(self.tokens_to_keep3D, dtype=tf.bool), tf.cast(annotated3D, dtype=tf.bool)), dtype=tf.float32)
      #tokens_to_keep3D = tf.Print(tokens_to_keep3D, [tf.count_nonzero(tokens_to_keep3D, dtype=tf.float32)], "final tokenstokeep")

    else:
      tokens_to_keep3D = self.tokens_to_keep3D

    # need to repeat each of these once for each target in the sentence
    # mask = tf.gather_nd(tf.tile(tf.transpose(self.tokens_to_keep3D, [0, 2, 1]), [1, bucket_size, 1]),
    #                     tf.where(tf.equal(trigger_predictions, 1)))
    mask_tiled = tf.reshape(tf.tile(tf.squeeze(tokens_to_keep3D, -1), [1, bucket_size]), [batch_size, bucket_size, bucket_size])
    mask = tf.gather_nd(mask_tiled, tf.where(tf.equal(trigger_predictions, 1)))
    #mask = tf.Print(mask, [tf.shape(mask)], "mask shape")
    count = tf.cast(tf.count_nonzero(mask), tf.float32)

    # now we have k sets of targets for the k frames
    # (t1) f1 f2 f3
    # (t2) f1 f2 f3
    # srl_targets = targets[:, :, 3:]

    # get all the tags for each token (which is the trigger for a frame), structuring
    # targets3D as follows (assuming t1 and t2 are triggers for f1 and f3, respectively):
    # (t1) f1 f1 f1
    # (t2) f3 f3 f3
    srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

    # num_triggers_in_batch x seq_len
    predictions = tf.cast(tf.argmax(logits_transposed, axis=-1), tf.int32)

    #mask = tf.Print(mask, [tf.shape(mask), tf.shape(predictions)], "mask and predictions")
    # if annotated3D is not None:
    #   pass
    #   #predictions = tf.Print(predictions, [tf.shape(predictions)], "VN Predictions shape")

    trigger_counts = tf.reduce_sum(trigger_predictions, -1)

    def compute_srl_loss(logits_transposed, srl_targets_transposed, transition_params, mask):
      # batch*num_targets x seq_len -- num_triggers?
      seq_mask = tf.sequence_mask(tf.reshape(trigger_counts, [-1]))
      srl_targets_indices = tf.where(seq_mask)

      #srl_targets_indices = tf.Print(srl_targets_indices, [batch_size, bucket_size, tf.shape(logits_transposed), tf.shape(srl_targets_indices), seq_mask[0], srl_targets_indices[0]], summarize=200)

      # num_triggers_in_batch x seq_len
      srl_targets = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

      if ignore_label_idx is not None:
        preds_to_keep = tf.cast(tf.reduce_all(tf.not_equal(srl_targets, ignore_label_idx), axis=-1, keepdims=True), tf.float32)
        preds_to_keep = tf.Print(preds_to_keep, [tf.shape(preds_to_keep), tf.shape(annotated3D), tf.shape(mask), tf.shape(srl_targets)])
        mask = tf.multiply(mask, preds_to_keep)
      else:
        preds_to_keep = tf.ones_like(tf.reduce_max(srl_targets, axis=-1))

      if transition_params is not None:
        seq_lens = tf.reduce_sum(mask, 1)
        # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits_transposed, srl_targets,
                                                                              seq_lens,
                                                                              transition_params=transition_params)
        loss = tf.reduce_mean(-log_likelihood)
      else:
        if self.label_smoothing > 0:
          srl_targets_onehot = tf.one_hot(indices=srl_targets, depth=num_labels, axis=-1)
          # srl_targets_onehot = tf.Print(srl_targets_onehot, [tf.reduce_sum(trigger_counts), tf.shape(logits_transposed), tf.shape(srl_targets), tf.shape(srl_targets_onehot)], "srl logits", summarize=200)
          # srl_targets_onehot = tf.Print(srl_targets_onehot, [logits_transposed], "srl logits", summarize=200)
          #
          #srl_targets_onehot = tf.Print(srl_targets_onehot, [num_labels, tf.shape(srl_targets_onehot)], "srl targets", summarize=200)
          #srl_targets_onehot = tf.Print(srl_targets_onehot, [tf.shape(srl_targets_onehot)], "srl targets onehot", summarize=200)

          # orig_logits_shape = tf.shape(logits_transposed)

          # if annotated3D is not None:
          #   new_mask = mask
          #   #num_labels = tf.Print(num_labels, [num_labels, tf.shape(srl_targets_onehot)], "num labels")
          #   # todo don't hardcode num_labels
          #   class_weights_np = np.ones(50, dtype=np.float32)
          #   class_weights_np[2] = 0
          #   class_weights_np[0] = 0
          #   #class_weights_np[7] = 0
          #   class_weights = tf.convert_to_tensor(class_weights_np)
          #
          #   srl_targets_onehot = tf.Print(srl_targets_onehot, [tf.shape(srl_targets_onehot)], "srl_targets_onehot")
          #
          #   # preds_in_batch x bucket_size x num_labels
          #   sample_weights = tf.reduce_sum(tf.multiply(srl_targets_onehot, class_weights), 2)
          #
          #   sample_weights = tf.Print(sample_weights, [tf.shape(sample_weights)], "sample_weights")
          #
          #
          #   #sample_weights = tf.Print(sample_weights, [tf.shape(sample_weights), sample_weights], "class weights", summarize=10)
          #   new_mask = tf.cast(tf.logical_and(tf.cast(sample_weights, tf.bool), tf.cast(mask, tf.bool)), tf.float32)
          # else:
          #   new_mask = mask

          # Backprop into logits?
          loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(logits_transposed, [-1, num_labels]),
                                                 onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                 weights=tf.reshape(mask, [-1]),
                                                 label_smoothing=self.label_smoothing,
                                                 reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
          # cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cross entropy", summarize=200)
          # cross_entropy = tf.reshape(cross_entropy, [orig_logits_shape[0], orig_logits_shape[1]])

        else:
          cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_transposed, labels=srl_targets)
          cross_entropy *= mask
          loss = tf.cond(tf.equal(count, 0.), lambda: tf.constant(0.), lambda: tf.reduce_sum(cross_entropy) / count)

      if annotated3D is not None:
        preds_to_keep = tf.multiply(preds_to_keep, tf.cast(tf.reduce_any(tf.cast(mask, tf.bool), axis=-1, keepdims=True), tf.float32))
        # preds_to_keep = tf.Print(preds_to_keep, [tf.shape(preds_to_keep), tf.shape(annotated3D), tf.shape(mask)])
        #srl_targets = tf.Print(srl_targets, [tf.shape(srl_targets)], "VN Targets shape ")

      #predictions = tf.cast(tf.argmax(logits_transposed, axis=-1), tf.int32)
      #predictions = tf.Print(predictions, [tf.shape(predictions, tf.shape(srl_targets))], "predictions and targets")
      correct_unmasked = tf.cast(tf.equal(predictions, srl_targets), tf.float32)
      correct_masked = correct_unmasked * mask
      correct = tf.reduce_sum(correct_masked)
      return loss, correct, preds_to_keep, srl_targets

    # compute_loss = tf.logical_and(tf.greater(tf.shape(targets)[2], 0), tf.greater(tf.reduce_sum(trigger_counts), 0))
    # loss, correct, preds_to_ignore = tf.cond(compute_loss,
    #                lambda: compute_srl_loss(logits_transposed, srl_targets_transposed, transition_params, mask),
    #                lambda: (tf.constant(0.), tf.constant(0.), tf.constant(0.)))

    loss, correct, preds_to_keep, targets = compute_srl_loss(logits_transposed, srl_targets_transposed, transition_params, mask)

    probabilities = tf.nn.softmax(logits_transposed)
    if annotated3D is not None:
      pass
      #correct = tf.Print(correct, [correct], "VN correct ")
      #count = tf.Print(count, [count], "VN count ")

    output = {
      'loss': loss,
      'probabilities': probabilities,
      'predictions': predictions,
      'logits': logits_transposed,
      'transition_params': tf.constant(0.),
      'count': count,
      'correct': correct,
      'preds_to_keep': preds_to_keep,
      'targets': targets
    }

    return output

  def output_predicates(self, logits, predicate_targets, predicate_true_start_idx):
    """"""

    # logits are batch x seq_len x 2

    # original_shape = tf.shape(logits)
    # batch_size = original_shape[0]
    # bucket_size = original_shape[1]

    # now we have k sets of targets for the k frames
    # (t1) f1 f2 v0
    # (t2) v0 f2 f3
    # (t3) f1 f2 f3
    # (t4) f1 v0 f3
    # srl_targets = targets[:,:,3:]

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=predicate_targets)

    cross_entropy *= tf.squeeze(self.tokens_to_keep3D, -1)
    loss = tf.reduce_sum(cross_entropy) / self.n_tokens

    squeezed_mask = tf.squeeze(self.tokens_to_keep3D, -1)
    int_mask = tf.cast(squeezed_mask, tf.int32)

    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.equal(predictions, predicate_targets), tf.float32) * squeezed_mask)
    predictions *= int_mask

    predicate_predictions = tf.where(tf.greater(predictions, predicate_true_start_idx), tf.ones_like(predictions), tf.zeros_like(predictions))
    predicate_predictions *= int_mask

    output = {
      'loss': loss,
      'predicate_predictions': predicate_predictions,
      'predictions': predictions * tf.cast(tf.squeeze(self.tokens_to_keep3D, -1), tf.int32),
      'logits': logits,
      # 'gold_trigger_predictions': tf.transpose(predictions, [0, 2, 1]),
      'count': self.n_tokens,
      'correct': correct,
      'targets': predicate_targets,
    }

    return output

  # =============================================================
  def output_svd(self, logits3D, targets3D):
    """"""
    targets_shape = tf.shape(targets3D)
    batch_size = targets_shape[0]
    bucket_size = targets_shape[1]
    original_shape = [batch_size, bucket_size, bucket_size]
    flat_shape = tf.stack([batch_size, bucket_size])
    tokens_to_keep1D = tf.reshape(self.tokens_to_keep3D, [-1])
    targets1D = tf.reshape(targets3D, [-1])

    def dummy_loss():
      # log_loss, roots_loss, pairs_log_loss, svd_loss, n_cycles, len_2_cycles, probabilities, predictions, correct
      return tf.constant(0.), tf.constant(0.), tf.constant(0.), tf.constant(0.), tf.constant(-1.), tf.constant(-1.), \
             tf.zeros(original_shape), tf.zeros(flat_shape, dtype=tf.int32), tf.constant(0.)

    def compute_loss(logits3D, tokens_to_keep1D):

      # flatten to [B*N, N]
      logits2D = tf.reshape(logits3D, tf.stack([batch_size * bucket_size, -1]))
      targets_mask = self.gen_targets_mask(targets3D, batch_size, bucket_size)

      ######## pairs softmax thing #########
      pairs_log_loss, pairs_concat = tf.cond(tf.equal(self.pairs_penalty, tf.constant(0.0)),
                           lambda: (tf.constant(0.0), tf.concat([tf.transpose(tf.expand_dims(logits3D, -1), [0, 2, 1, 3]), tf.expand_dims(logits3D, -1)], axis=-1)),
                           lambda: self.compute_pairs_loss(logits3D, targets_mask, batch_size, bucket_size))

      ######### roots loss (diag) ##########
      roots_loss = tf.cond(tf.equal(self.roots_penalty, tf.constant(0.0)),
                         lambda: tf.constant(0.0),
                         lambda: self.compute_roots_loss(logits3D, targets_mask))

      ########## normal log loss ##########
      cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2D, labels=targets1D)
      log_loss = tf.reduce_sum(cross_entropy1D * tokens_to_keep1D) / self.n_tokens

      ########## pairs mask #########
      logits3D = tf.cond(tf.constant(self.mask_pairs),
                         lambda: self.logits_mask_pairs(logits3D, batch_size),
                         lambda: logits3D)

      ########## roots mask (diag) #########
      logits3D = tf.cond(tf.constant(self.mask_roots),
                         lambda: self.logits_mask_roots(logits3D, batch_size, bucket_size),
                         lambda: logits3D)

      mask = (1 - self.tokens_to_keep3D) * -(tf.abs(tf.reduce_min(logits3D)) + tf.abs(tf.reduce_max(logits3D)))
      logits3D_masked = logits3D + mask
      logits3D_masked = tf.transpose(mask, [0, 2, 1]) + logits3D_masked
      logits3D = logits3D_masked

      logits2D = tf.reshape(logits3D, tf.stack([batch_size * bucket_size, -1]))
      predictions1D = tf.to_int32(tf.argmax(logits2D, 1))
      probabilities2D = tf.nn.softmax(logits2D)
      correct1D = tf.to_float(tf.equal(predictions1D, targets1D))

      ########### svd loss ##########
      svd_loss = tf.cond(tf.equal(self.svd_penalty, tf.constant(0.0)),
                         lambda: tf.constant(0.0),
                         lambda: self.compute_svd_loss(logits2D, tokens_to_keep1D, batch_size, bucket_size))

      # at test time
      # if self.moving_params is not None and self.svd_tree:
      if self.svd_tree:
        n_cycles, len_2_cycles = self.compute_cycles(logits2D, self.tokens_to_keep3D, batch_size, bucket_size)
      else:
        n_cycles = len_2_cycles = tf.constant(-1.)

      return log_loss, roots_loss, pairs_log_loss, svd_loss, n_cycles, len_2_cycles, \
             tf.reshape(probabilities2D, original_shape), tf.reshape(predictions1D, flat_shape), \
             correct1D * tokens_to_keep1D,

    log_loss, roots_loss, pairs_log_loss, svd_loss, n_cycles, len_2_cycles, probabilities, predictions, correct = tf.cond(
      tf.greater(tf.rank(logits3D), 1),
      lambda: compute_loss(logits3D, tokens_to_keep1D),
      lambda: dummy_loss())

    n_correct = tf.reduce_sum(correct * tokens_to_keep1D)
    accuracy = n_correct / self.n_tokens

    loss = log_loss + roots_loss + pairs_log_loss + svd_loss

    output = {
      'probabilities': probabilities,
      'predictions': predictions,
      'tokens': tokens_to_keep1D,
      'correct': correct,
      'n_correct': n_correct,
      'n_tokens': self.n_tokens,
      'accuracy': accuracy,
      'loss': loss,
      'log_loss': log_loss,
      'roots_loss': roots_loss,
      '2cycle_loss': pairs_log_loss,
      'svd_loss': svd_loss,
      'n_cycles': n_cycles,
      'len_2_cycles': len_2_cycles
    }

    return output


  ########## roots mask (diag) #########
  # select roots using softmax over diag. do this by choosing the root, then setting everything else
  # on diag to -1e9
  def logits_mask_roots(self, logits3D, batch_size, bucket_size):
    diag_mask = 1 - tf.eye(bucket_size, batch_shape=[batch_size])
    roots_logits = tf.matrix_diag_part(logits3D)  # doing this again with pairs mask applied
    idx_t = tf.cast(tf.argmax(roots_logits, axis=1), tf.int32)
    idx = tf.stack([tf.range(batch_size), idx_t], axis=-1)
    diagonal = tf.scatter_nd(idx, tf.fill([batch_size], tf.reduce_max(logits3D) + 1), [batch_size, bucket_size])
    diag_inv_mask = 1 - tf.scatter_nd(idx, tf.ones([batch_size]), [batch_size, bucket_size])
    diagonal_masked = diagonal + (tf.reduce_min(logits3D) - 1) * diag_inv_mask
    roots_mask = tf.matrix_diag(diagonal_masked)
    return roots_mask + diag_mask * logits3D


  ########## pairs mask #########
  def logits_mask_pairs(self, logits3D, batch_size):
    logits_expanded = tf.expand_dims(logits3D, -1)
    pairs_concat = tf.concat([tf.transpose(logits_expanded, [0, 2, 1, 3]), logits_expanded], axis=-1)
    maxes = tf.reduce_max(pairs_concat, axis=-1)
    # min across each sequence in the batch
    min_vals = tf.reshape(tf.reduce_min(tf.reshape(logits3D, [batch_size, -1]), axis=-1), [batch_size, 1, 1])
    mask_eq = tf.cast(tf.equal(maxes, logits3D), tf.float32)
    mask_neq = tf.cast(tf.not_equal(maxes, logits3D), tf.float32)
    return logits3D * mask_eq + mask_neq * min_vals


  # targets_mask is the target adjacency matrix (with zero padding)
  def gen_targets_mask(self, targets3D, batch_size, bucket_size):
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    idx = tf.stack([i1, i2, targets3D], axis=-1)
    targets_mask = tf.scatter_nd(idx, tf.ones([batch_size, bucket_size]), [batch_size, bucket_size, bucket_size])
    targets_mask *= self.tokens_to_keep3D

    # assert that there is exactly one root
    with tf.control_dependencies(
            [tf.assert_equal(tf.reduce_sum(tf.matrix_diag_part(targets_mask), axis=1), tf.ones([batch_size]))]):
      return targets_mask


  ########### svd loss ##########
  def compute_svd_loss(self, logits2D, tokens_to_keep1D, batch_size, bucket_size):
    # construct predicted adjacency matrix
    maxes = tf.expand_dims(tf.reduce_max(logits2D, axis=1), 1)
    maxes_tiled = tf.tile(maxes, [1, bucket_size])
    adj_flat = tf.cast(tf.equal(logits2D, maxes_tiled), tf.float32)
    adj_flat = adj_flat * tf.expand_dims(tokens_to_keep1D, -1)
    adj = tf.reshape(adj_flat, [batch_size, bucket_size, bucket_size])
    # zero out diagonal
    adj = tf.matrix_set_diag(adj, tf.zeros([batch_size, bucket_size]))
    # make it undirected
    undirected_adj = tf.cast(tf.logical_or(tf.cast(adj, tf.bool), tf.transpose(tf.cast(adj, tf.bool), [0, 2, 1])), tf.float32)

    # compute laplacian & its trace
    degrees = tf.reduce_sum(undirected_adj, axis=1)
    l_trace = tf.reduce_sum(degrees, axis=1)
    laplacian = tf.matrix_set_diag(-undirected_adj, degrees)

    svd_loss = 0.
    try:
      dtype = laplacian.dtype
      _, s, _ = tf.py_func(np.linalg.svd, [laplacian, False, True], [dtype, dtype, dtype])
      # s, _, _ = tf.svd(laplacian)
      l_rank = tf.reduce_sum(tf.cast(tf.greater(s, 1e-15), tf.float32), axis=1)

      # cycles iff: 0.5 * l_trace > l_rank + 1
      svd_loss = tf.maximum(0.5 * l_trace - (l_rank + 1), tf.constant(0.0))
      # svd_loss_masked = self.tokens_to_keep3D * svd_loss
      svd_loss = self.svd_penalty * tf.reduce_sum(svd_loss)  # / self.n_tokens
    except np.linalg.linalg.LinAlgError:
      print("SVD did not converge")
    return svd_loss


  ########### cycles ##########
  def compute_cycles(self, logits2D_masked, tokens_to_keep3D, batch_size, bucket_size):
    # construct predicted adjacency matrix

    # # max values for every token (flattened across batches)
    # # mask = (1 - tokens_to_keep3D) * -(tf.abs(tf.reduce_min(logits3D)) + tf.abs(tf.reduce_max(logits3D)))
    # # logits3D_masked = logits3D + mask
    # # logits3D_masked = tf.transpose(mask, [0, 2, 1]) + logits3D_masked
    # # logits2D_masked = tf.reshape(logits3D_masked, [batch_size * bucket_size, -1])
    # maxes = tf.expand_dims(tf.reduce_max(logits2D_masked, axis=1), -1)
    # # tile the maxes across rows
    # maxes_tiled = tf.tile(maxes, [1, bucket_size])
    # # 1 where logits2d == max, 0 elsewhere
    # adj_flat = tf.cast(tf.equal(logits2D_masked, maxes_tiled), tf.float32)
    # # zero out padding
    # adj_flat = adj_flat * tf.reshape(tokens_to_keep3D, [-1, 1])
    # # reshape into [batch, bucket, bucket]
    # adj = tf.reshape(adj_flat, [batch_size, bucket_size, bucket_size])
    max_vals = tf.cast(tf.argmax(tf.reshape(logits2D_masked, [batch_size, bucket_size, bucket_size]), axis=2), tf.int32)
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    idx = tf.stack([i1, i2, max_vals], axis=-1)
    adj = tf.scatter_nd(idx, tf.ones([batch_size, bucket_size]), [batch_size, bucket_size, bucket_size])
    adj = adj * tokens_to_keep3D



    # zero out diagonal
    adj = tf.matrix_set_diag(adj, tf.zeros([batch_size, bucket_size]))
    # make it undirected
    undirected_adj = tf.cast(tf.logical_or(tf.cast(adj, tf.bool), tf.cast(tf.transpose(adj, [0, 2, 1]) * tokens_to_keep3D, tf.bool)), tf.float32)

    # compute laplacian & its trace
    degrees = tf.reduce_sum(undirected_adj, axis=1)
    l_trace = tf.reduce_sum(degrees, axis=1)
    laplacian = tf.matrix_set_diag(-undirected_adj, degrees)

    # 1 where i->j and j->i are both set in adj
    pairs = tf.multiply(adj, tf.transpose(adj, [0, 2, 1]))
    len_2_cycles = tf.greater(tf.reduce_sum(tf.reshape(pairs, [batch_size, -1]), axis=-1), tf.constant(0.))

    # svd_loss = 0.
    # try:
    # s = tf.py_func(np.linalg.svd, [laplacian, False, False], [laplacian.dtype])
    with tf.device('/cpu:0'):
      s = tf.svd(laplacian, compute_uv=False)

    # this is the tol used in numpy.linalg.matrix_rank
    # tol = tf.reduce_max(s) * tf.cast(tf.reduce_max(tf.shape(laplacian)), tf.float32) * np.finfo(np.float32).eps

    # this is what matlab does (maybe the numpy one is more suitable for QR? idk)
    tol = tf.cast(tf.reduce_max(tf.shape(laplacian)), tf.float32) * float32_eps

    l_rank = tf.reduce_sum(tf.cast(tf.greater(s, tol), tf.float32), axis=1)

    # cycles iff: 0.5 * l_trace >= l_rank + 1
    n_cycles = tf.greater_equal(0.5 * l_trace, l_rank + 1)

    # svd_loss = tf.maximum(0.5 * l_trace - (l_rank + 1), tf.constant(0.0))
    # svd_loss_masked = self.tokens_to_keep3D * svd_loss
    # svd_loss = self.svd_penalty * tf.reduce_sum(svd_loss)  # / self.n_tokens
    # except np.linalg.linalg.LinAlgError:
    #   print("SVD did not converge")

    # n_cycles = tf.cond(tf.equal(l_trace, 70), lambda: tf.Print(n_cycles, [s], "eigenvalues", summarize=50), lambda: n_cycles)

    # ij_0 = (tf.constant(0), n_cycles)
    # c = lambda i, j: i < batch_size
    # b = lambda i, j: tf.cond(tf.equal(l_trace[i], 84), lambda: (i+1, tf.Print(n_cycles, [s[i], l_rank[i], tol], "eigenvalues", summarize=50)), lambda: (i+1, n_cycles))
    # _, n_cycles = tf.while_loop(c, b, ij_0)

    # n_cycles = tf.Print(n_cycles, [tf.reduce_sum(tf.cast(n_cycles, tf.int32))], "n_cycles in batch", summarize=50)
    # len_2_cycles = tf.Print(len_2_cycles, [tf.reduce_sum(tf.cast(len_2_cycles, tf.int32))], "len_2_cycles in batch", summarize=50)

    # logits3D = tf.reshape(logits2D_masked, [batch_size, bucket_size, bucket_size])
    # ij_0 = (tf.constant(0), n_cycles)
    # c = lambda i, j: i < batch_size
    # b = lambda i, j: tf.cond(tf.logical_or(n_cycles[i], len_2_cycles[i]), lambda: (i + 1, tf.Print(n_cycles, [adj[i], logits3D[i]], summarize=10000)), lambda: (i + 1, n_cycles))
    # _, n_cycles = tf.while_loop(c, b, ij_0)

    return n_cycles, len_2_cycles


  ######### roots loss (diag) ##########
  def compute_roots_loss(self, logits3D, targets_mask):
    # softmax over diagonal
    roots_logits = tf.matrix_diag_part(logits3D)
    roots_targets1D = tf.argmax(tf.matrix_diag_part(targets_mask), axis=1)
    roots_cross_entropy1D = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=roots_logits, labels=roots_targets1D)
    return self.roots_penalty * tf.reduce_mean(roots_cross_entropy1D)


  ######## pairs softmax thing #########
  # concat preds and preds^T along last (new) dim. flatten. now do softmax loss, forcing only one of each
  # pair to be predicted, enforcing no cycles between two nodes.
  def compute_pairs_loss(self, logits3D, targets_mask, batch_size, bucket_size):

    # this has 1s in all the locations of the adjacency matrix that we care about: i,j and j,i where i,j is correct
    # add is ok because we know that no two will ever be set (except diag which we zero out anyway)
    diag_mask = 1 - tf.eye(bucket_size, batch_shape=[batch_size])
    pairs_mask = tf.add(targets_mask, tf.transpose(targets_mask, [0, 2, 1])) * diag_mask

    targets_mask1D = tf.reshape(targets_mask, [-1])
    logits_expanded = tf.expand_dims(logits3D, -1)
    pairs_concat = tf.concat([tf.transpose(logits_expanded, [0, 2, 1, 3]), logits_expanded], axis=-1)
    pairs_logits2D = tf.reshape(pairs_concat, [batch_size * bucket_size * bucket_size, 2])
    pairs_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pairs_logits2D,
                                                                labels=tf.cast(targets_mask1D, tf.int32))
    pairs_xent3D = tf.reshape(pairs_xent, [batch_size, bucket_size, bucket_size])
    return self.pairs_penalty * tf.reduce_sum(pairs_xent3D * pairs_mask) / self.n_tokens, pairs_concat

  #=============================================================
  def conditional_probabilities(self, logits4D, transpose=True):
    """"""
    
    if transpose:
      logits4D = tf.transpose(logits4D, [0,1,3,2])
    original_shape = tf.shape(logits4D)
    n_classes = original_shape[3]
    
    logits2D = tf.reshape(logits4D, tf.stack([-1, n_classes]))
    probabilities2D = tf.nn.softmax(logits2D)
    return tf.reshape(probabilities2D, original_shape)
  
  #=============================================================
  def tag_argmax(self, tag_probs, tokens_to_keep):
    """"""
    
    return np.argmax(tag_probs[:,Vocab.ROOT:], axis=1)+Vocab.ROOT

  def get_sample_prob(self, step):
    if self.sampling_schedule == 'constant':
      return tf.constant(self.sample_prob)
    if self.sampling_schedule == 'sigmoid':
      #   y = k / (k + np.exp(x / k))
      return self.sample_prob/(self.sample_prob+tf.exp(tf.multiply(step, (1/self.sample_prob))))

  # =============================================================
  def check_cycles_svd(self, parse_preds, length):

    # tokens_to_keep[0] = True
    # length = np.sum(tokens_to_keep)
    # I = np.eye(len(tokens_to_keep))
    # # block loops and pad heads
    # parse_probs = parse_probs * tokens_to_keep * (1 - I)
    # parse_preds = np.argmax(parse_probs, axis=1)
    # tokens = np.arange(1, length)
    # roots = np.where(parse_preds[tokens] == 0)[0] + 1
    #

    # print("parse preds")
    # print(parse_preds)

    laplacian = np.zeros((length, length))
    for i, p in enumerate(parse_preds[:length]):
      # print(p)
      # if p != 0:
      if i != p:
        laplacian[i, p] = -1.
        laplacian[p, i] = -1.

    degrees = -np.sum(laplacian, axis=0)
    for i, d in enumerate(degrees):
      laplacian[i, i] = d

    e = scipy.linalg.svd(laplacian, compute_uv=False)
    rank = np.sum(np.greater(e, 1e-15))

    adj = np.zeros((len(parse_preds), len(parse_preds)))
    for i, p in enumerate(parse_preds):  # [1:length]):
      if i != p:
        adj[i, p] = 1

    len_2_cycles = np.sum(np.multiply(adj, np.transpose(adj))) > 0

    # has_cycle = len_2_cycles or (0.5 * np.trace(laplacian) >= rank + 1)
    return int(len_2_cycles), int((0.5 * np.trace(laplacian) >= rank + 1))

  # ensure at least one root
  def ensure_gt_one_root(self, parse_preds, parse_probs, tokens):
    # The current root probabilities
    root_probs = np.diagonal(parse_probs[tokens])
    # The current head probabilities
    old_head_probs = parse_probs[tokens, parse_preds[tokens]]
    # Get new potential root probabilities
    new_root_probs = root_probs / old_head_probs
    # Select the most probable root
    new_root = tokens[np.argmax(new_root_probs)]
    # Make the change
    parse_preds[new_root] = new_root
    return parse_preds

  # ensure at most one root
  def ensure_lt_two_root(self, parse_preds, parse_probs, roots, tokens):
    # The probabilities of the current heads
    root_probs = parse_probs[roots, roots]
    # Set the probability of depending on the root zero
    parse_probs[roots, roots] = 0
    # Get new potential heads and their probabilities
    new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) # + 1
    new_head_probs = parse_probs[roots, new_heads] / root_probs
    # Select the most probable root
    new_root = roots[np.argmin(new_head_probs)]
    # Make the change
    parse_preds[roots] = new_heads
    parse_preds[new_root] = new_root
    return parse_preds

  #=============================================================
  def parse_argmax(self, parse_probs, tokens_to_keep, n_cycles=-1, len_2_cycles=-1):
    """"""
    # tokens_to_keep[0] = True
    length = np.sum(tokens_to_keep)
    # tokens = np.arange(1, length)
    tokens = np.arange(length)
    parse_probs = parse_probs * tokens_to_keep
    parse_preds = np.argmax(parse_probs, axis=1)
    roots = [i for i, p in enumerate(parse_preds[:length]) if i == p]
    num_roots = len(roots)
    roots_lt = 1. if num_roots < 1 else 0.
    roots_gt = 1. if num_roots > 1 else 0.
    # len_2_cycles = 0
    # n_cycles = 0
    # ensure_tree = self.ensure_tree

    # todo this should really happen in model, in batch, and be passed in
    # if self.svd_tree:
      # n_cycles = cycle
      # len_2_cycles, n_cycles = self.check_cycles_svd(parse_preds, length)

    if self.ensure_tree:
      if roots_lt:
        parse_preds = self.ensure_gt_one_root(parse_preds, parse_probs, tokens)
      elif roots_gt:
        parse_preds = self.ensure_lt_two_root(parse_preds, parse_probs, roots, tokens)

      # root_probs = np.diag(parse_probs)
      # parse_probs_no_roots = parse_probs * (1 - np.eye(parse_probs.shape[0]))
      # parse_probs_roots_aug = np.hstack([np.expand_dims(root_probs, -1), parse_probs_no_roots])
      # # parse_probs_roots_aug = np.vstack([np.zeros(parse_probs.shape[0]+1), parse_probs_roots_aug])
      # parse_preds_roots_aug = np.argmax(parse_probs_roots_aug, axis=1)
      #
      # np.set_printoptions(threshold=np.nan)
      #
      # coo = scipy.sparse.coo_matrix((np.ones(length), (np.arange(1, length + 1), parse_preds_roots_aug[:length])),
      #                               shape=(length + 1, length + 1))
      coo2 = scipy.sparse.coo_matrix((np.ones(length), (tokens, parse_preds[:length])), shape=(length, length))
      cc_count, ccs = scipy.sparse.csgraph.connected_components(coo2, directed=True, connection='weak', return_labels=True)
      if (cc_count > 1 and( n_cycles == 0 and len_2_cycles == 0)) or (cc_count == 1 and (n_cycles > 0 or len_2_cycles > 0)):
        # _, sizes = np.unique(ccs, return_counts=True)
        # len_2_cycles_tar = np.any(sizes == 2)
        # n_cycles_tar = np.any(sizes != 2)
        print("SVD: len_2_cycles: %d; n_cycles: %d" % (len_2_cycles, n_cycles))
        print("Tarjan: n_cycles: %d" % (cc_count > 1))
        print("labels: ", ccs)
        # a = coo.toarray()
        # for r in a:
        #   print(' '.join(map(str, r)))

        a = coo2.toarray()
        #for r in a:
        #  print("[%s]," % ', '.join(map(str, r)))

      if not self.svd_tree or len_2_cycles or n_cycles:
        root_probs = np.diag(parse_probs)
        parse_probs_no_roots = parse_probs * (1 - np.eye(parse_probs.shape[0]))
        parse_probs_roots_aug = np.hstack([np.expand_dims(root_probs, -1), parse_probs_no_roots])
        parse_probs_roots_aug = np.vstack([np.zeros(parse_probs.shape[0]+1), parse_probs_roots_aug])
        mst = scipy.sparse.csgraph.minimum_spanning_tree(-parse_probs_roots_aug)
        mst_arr = mst.toarray()[1:length+1]
        roots = mst_arr[:,0]
        mst_arr = mst_arr[:,1:length+1]
        mst_arr[tokens, tokens] = roots
        parse_preds = np.argmin(mst_arr, axis=1)

        # print("tree parse preds", parse_preds)



    # # if ensure_tree:
    #   len_2_cycles = n_cycles = 0
    #   # remove cycles
    #   # parse_probs_no_roots = parse_probs * (1 - np.eye(len(tokens_to_keep)))
    #   parse_preds_roots_aug = np.argmax(parse_probs_roots_aug, axis=1)
    #   tarjan = Tarjan(parse_preds_roots_aug, np.arange(1, length))
    #   cycles = tarjan.SCCs
    #   for SCC in cycles:
    #     if len(SCC) > 1:
    #       if len(SCC) == 2:
    #         len_2_cycles += 1.
    #       else:
    #         n_cycles += 1.
    #       dependents = set()
    #       to_visit = set(SCC)
    #       while len(to_visit) > 0:
    #         node = to_visit.pop()
    #         if not node in dependents:
    #           dependents.add(node)
    #           to_visit.update(tarjan.edges[node])
    #       # The indices of the nodes that participate in the cycle
    #       cycle = np.array(list(SCC))
    #       # The probabilities of the current heads
    #       old_heads = parse_preds_roots_aug[cycle]
    #       old_head_probs = parse_probs_roots_aug[cycle, old_heads]
    #       # Set the probability of depending on a non-head to zero
    #       non_heads = np.array(list(dependents))
    #       parse_probs_roots_aug[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
    #       # Get new potential heads and their probabilities
    #       new_heads = np.argmax(parse_probs_roots_aug[cycle][:, tokens], axis=1) + 1
    #       new_head_probs = parse_probs_roots_aug[cycle, new_heads] / old_head_probs
    #       # Select the most probable change
    #       change = np.argmax(new_head_probs)
    #       changed_cycle = cycle[change]
    #       old_head = old_heads[change]
    #       new_head = new_heads[change]
    #       # Make the change
    #       parse_preds_roots_aug[changed_cycle] = new_head
    #       tarjan.edges[new_head].add(changed_cycle)
    #       tarjan.edges[old_head].remove(changed_cycle)
    #   parse_probs_roots_aug = parse_probs_roots_aug[:length]
    #   roots = parse_probs_roots_aug[:, 0]
    #   parse_probs_roots_aug = parse_probs_roots_aug[:, 1:length + 1]
    #   # print(parse_probs_roots_aug)
    #   parse_probs_roots_aug[np.arange(length), np.arange(length)] = roots
    #   parse_preds = np.argmin(parse_probs_roots_aug, axis=1)
    return parse_preds, roots_lt, roots_gt

  
  #=============================================================
  def rel_argmax(self, rel_probs, tokens_to_keep):
    """"""
    
    if self.ensure_tree:
      # tokens_to_keep[0] = True
      rel_probs[:,Vocab.PAD] = 0
      root = Vocab.ROOT
      length = np.sum(tokens_to_keep)
      tokens = np.arange(length)
      rel_preds = np.argmax(rel_probs, axis=1)
      roots = np.where(rel_preds[tokens] == root)[0] #+1
      if len(roots) < 1:
        # rel_preds[1+np.argmax(rel_probs[tokens,root])] = root
        rel_preds[np.argmax(rel_probs[tokens, root])] = root
      elif len(roots) > 1:
        root_probs = rel_probs[roots, root]
        rel_probs[roots, root] = 0
        new_rel_preds = np.argmax(rel_probs[roots], axis=1)
        new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
        new_root = roots[np.argmin(new_rel_probs)]
        rel_preds[roots] = new_rel_preds
        rel_preds[new_root] = root
      return rel_preds
    else:
      rel_probs[:,Vocab.PAD] = 0
      rel_preds = np.argmax(rel_probs, axis=1)
      return rel_preds
  
  #=============================================================
  def __call__(self, inputs, targets, moving_params=None):
    """"""
    
    raise NotImplementedError()
  
  #=============================================================
  # @property
  # def global_sigmoid(self):
  #   return self._global_sigmoid
