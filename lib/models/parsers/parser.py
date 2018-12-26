#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf

from lib.models import nn

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class Parser(BaseParser):
  """"""

  def print_once(self, *args, **kwargs):
    if self.print_stuff:
      print(*args, **kwargs)
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""

    self.print_stuff = dataset.name == "Trainset"
    self.multi_penalties = {k: float(v) for k, v in map(lambda s: s.split(':'), self.multitask_penalties.split(';'))} if self.multitask_penalties else {}
    self.multi_layers = {k: set(map(int, v.split(','))) for k, v in map(lambda s: s.split(':'), self.multitask_layers.split(';'))} if self.multitask_layers else {}

    # todo use variables for vocabs this indexing is stupid
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets

    shape = tf.shape(targets[:, :, 1])
    batch_size = shape[0]
    bucket_size = shape[1]

    # extract out pb and vn srl labels
    srl_targets_combined = targets[:, :, 3:]

    total_srl_counts = tf.count_nonzero(srl_targets_combined, axis=-1)
    max_preds_in_batch = tf.reduce_max(total_srl_counts) // 2

    pb_take_mask = tf.sequence_mask(total_srl_counts / 2, maxlen=tf.reduce_max(total_srl_counts))

    pb_indices = tf.where(pb_take_mask)
    vn_indices = tf.where(tf.logical_not(pb_take_mask))
    vn_scatter_indices = tf.where(tf.reverse(tf.logical_not(pb_take_mask), [-1]))

    srl_targets_nopad = tf.gather_nd(srl_targets_combined, pb_indices)
    vn_targets_nopad = tf.gather_nd(srl_targets_combined, vn_indices)
    #srl_targets_nopad = tf.Print(srl_targets_nopad, [tf.shape(vn_scatter_indices), tf.shape(vn_targets_nopad), tf.shape(pb_indices), tf.shape(srl_targets_nopad), vn_scatter_indices, pb_indices], summarize=100)

    srl_targets = tf.scatter_nd(pb_indices, srl_targets_nopad, [tf.cast(batch_size, tf.int64), tf.cast(bucket_size, tf.int64), max_preds_in_batch])
    vn_targets = tf.scatter_nd(vn_scatter_indices, vn_targets_nopad, [tf.cast(batch_size, tf.int64), tf.cast(bucket_size, tf.int64), max_preds_in_batch])
    #vn_targets = tf.ones_like(srl_targets)

    annotated = dataset.annotated
    step = dataset.step

    num_pos_classes = len(vocabs[1])
    num_rel_classes = len(vocabs[2])
    num_srl_classes = len(vocabs[3])
    num_pred_classes = len(vocabs[4])
    num_vn_classes = len(vocabs[6])

    # need to add batch dim for batch size 1
    #inputs = tf.Print(inputs, [tf.shape(inputs), tf.shape(targets)], summarize=10)

    reuse = (moving_params is not None)
    # Finds tokens which are not PAD or ROOT but includes UNK?
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    annotated_3D = tf.expand_dims(tf.to_float(annotated), 2)
    # annotated_3D = tf.Print(annotated_3D, [tf.shape(annotated_3D)], "annotated_3D")
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params

    vn_nolabel_idx = vocabs[6]['NoLabel'][0]
    print(vn_nolabel_idx)

    if self.add_to_pretrained:
      word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:, :, 0], inputs[:, :, 1],
                                                            moving_params=self.moving_params)
      word_inputs += pret_inputs
    else:
      word_inputs = vocabs[0].embedding_lookup(inputs[:, :, 1], moving_params=self.moving_params)
      #print('Word inputs: ', word_inputs.shape)
    if self.word_l2_reg > 0:
      unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)), 2)
      word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    inputs_to_embed = [word_inputs]
    if self.add_pos_to_input:
      pos_inputs = vocabs[1].embedding_lookup(inputs[:, :, 2], moving_params=self.moving_params)
      inputs_to_embed.append(pos_inputs)

    embed_inputs = self.embed_concat(*inputs_to_embed)

    if self.add_predicates_to_input:
      predicate_embed_inputs = vocabs[4].embedding_lookup(inputs[:, :, 3], moving_params=self.moving_params)
      embed_inputs = tf.concat([embed_inputs, predicate_embed_inputs], axis=2)
    
    top_recur = embed_inputs

    attn_weights_by_layer = {}
    combine_with_vn = self.combine_with_vn
    predict_vn = self.predict_vn

    hidden_size = self.num_heads * self.head_size
    self.print_once("n_recur: ", self.n_recur)
    self.print_once("num heads: ", self.num_heads)
    self.print_once("cnn dim: ", self.cnn_dim)
    self.print_once("relu hidden size: ", self.relu_hidden_size)
    self.print_once("head size: ", self.head_size)

    self.print_once("cnn2d_layers: ", self.cnn2d_layers)
    self.print_once("cnn_dim_2d: ", self.cnn_dim_2d)

    self.print_once("multitask penalties: ", self.multi_penalties)
    self.print_once("multitask layers: ", self.multi_layers)

    self.print_once("sampling schedule: ", self.sampling_schedule)


    # maps joint predicate/pos indices to pos indices
    preds_to_pos_map = np.zeros([num_pred_classes, 1], dtype=np.int32)
    if self.joint_pos_predicates:
      for pred_label, pred_idx in vocabs[4].iteritems():
        if pred_label in vocabs[4].SPECIAL_TOKENS:
          postag = pred_label
        else:
          _, postag = pred_label.split('/')
        pos_idx = vocabs[1][postag]
        preds_to_pos_map[pred_idx] = pos_idx

    # todo these are actually wrong because of nesting
    bilou_constraints = np.zeros((num_srl_classes, num_srl_classes))
    if self.transition_statistics:
      with open(self.transition_statistics, 'r') as f:
        for line in f:
          tag1, tag2, prob = line.split("\t")
          #print('Tags, prob: ', tag1, tag2, prob)
          bilou_constraints[vocabs[3][tag1], vocabs[3][tag2]] = float(prob)

    ###### stuff for multitask attention ######
    multitask_targets = {}

    mask2d = self.tokens_to_keep3D * tf.transpose(self.tokens_to_keep3D, [0, 2, 1])

    # compute targets adj matrix
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    idx = tf.stack([i1, i2, targets[:, :, 1]], axis=-1)
    adj = tf.scatter_nd(idx, tf.ones([batch_size, bucket_size]), [batch_size, bucket_size, bucket_size])
    adj = adj * mask2d

    # roots_mask = 1. - tf.expand_dims(tf.eye(bucket_size), 0)

    # create parents targets
    parents = targets[:, :, 1]
    multitask_targets['parents'] = parents

    # create children targets
    multitask_targets['children'] = parents

    # create grandparents targets
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    idx = tf.reshape(tf.stack([i1, tf.nn.relu(parents)], axis=-1), [-1, 2])
    grandparents = tf.reshape(tf.gather_nd(parents, idx), [batch_size, bucket_size])
    multitask_targets['grandparents'] = grandparents
    grand_idx = tf.stack([i1, i2, grandparents], axis=-1)
    grand_adj = tf.scatter_nd(grand_idx, tf.ones([batch_size, bucket_size]), [batch_size, bucket_size, bucket_size])
    grand_adj = grand_adj * mask2d

    # whether to condition on gold or predicted parse
    use_gold_parse = self.inject_manual_attn and not ((moving_params is not None) and self.gold_attn_at_train)
    sample_prob = self.get_sample_prob(step)
    if use_gold_parse and (moving_params is None):
      use_gold_parse_tensor = tf.less(tf.random_uniform([]), sample_prob)
    else:
      use_gold_parse_tensor = tf.equal(int(use_gold_parse), 1)

    print("use gold parse (%s): " % dataset.name, use_gold_parse)

    ##### Functions for predicting parse, Dozat-style #####
    def get_parse_logits(parse_inputs):
      if self.full_parse:
        ######## do parse-specific stuff (arcs) ########
        with tf.variable_scope('MLP', reuse=reuse):
          dep_mlp, head_mlp = self.MLP(parse_inputs, self.class_mlp_size + self.attn_mlp_size, n_splits=2)
          dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :self.attn_mlp_size], dep_mlp[:, :, self.attn_mlp_size:]
          head_arc_mlp, head_rel_mlp = head_mlp[:, :, :self.attn_mlp_size], head_mlp[:, :, self.attn_mlp_size:]

        with tf.variable_scope('Arcs', reuse=reuse):
          arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)

          arc_logits = tf.cond(tf.less_equal(tf.shape(tf.shape(arc_logits))[0], 2),
                               lambda: tf.reshape(arc_logits, [batch_size, 1, 1]), lambda: arc_logits)
          # arc_logits = tf.Print(arc_logits, [tf.shape(arc_logits), tf.shape(tf.shape(arc_logits))])
        return arc_logits, dep_rel_mlp, head_rel_mlp
      else:
        return dummy_parse_logits()

    def dummy_parse_logits():
      dummy_rel_mlp = tf.zeros([batch_size, bucket_size, self.class_mlp_size])
      return tf.zeros([batch_size, bucket_size, bucket_size]), dummy_rel_mlp, dummy_rel_mlp

    arc_logits, dep_rel_mlp, head_rel_mlp = dummy_parse_logits()

    ###########################################

    with tf.variable_scope("crf", reuse=reuse):  # to share parameters, change scope here
      if self.viterbi_train:
        transition_params = tf.get_variable("transitions", [num_srl_classes, num_srl_classes], initializer=tf.constant_initializer(bilou_constraints))
      elif self.viterbi_decode:
        transition_params = tf.get_variable("transitions", [num_srl_classes, num_srl_classes], initializer=tf.constant_initializer(bilou_constraints), trainable=False)
      else:
        transition_params = None
    #self.print_once("using transition params: ", transition_params)

    assert (self.cnn_layers != 0 and self.n_recur != 0) or self.num_blocks == 1, "num_blocks should be 1 if cnn_layers or n_recur is 0"
    assert self.dist_model == 'bilstm' or self.dist_model == 'transformer', 'Model must be either "transformer" or "bilstm"'

    for b in range(self.num_blocks):
      with tf.variable_scope("block%d" % b, reuse=reuse):  # to share parameters, change scope here
        # Project for CNN input
        if self.cnn_layers > 0:
          with tf.variable_scope('proj0', reuse=reuse):
            top_recur = self.MLP(top_recur, self.cnn_dim, n_splits=1)

        ####### 1D CNN ########
        with tf.variable_scope('CNN', reuse=reuse):
          kernel = 3
          for i in xrange(self.cnn_layers):
            with tf.variable_scope('layer%d' % i, reuse=reuse):
              if self.cnn_residual:
                top_recur += self.CNN(top_recur, 1, kernel, self.cnn_dim, self.recur_keep_prob, self.info_func)
                top_recur = nn.layer_norm(top_recur, reuse)
              else:
                top_recur = self.CNN(top_recur, 1, kernel, self.cnn_dim, self.recur_keep_prob, self.info_func)
          if self.cnn_residual and self.n_recur > 0:
            top_recur = nn.layer_norm(top_recur, reuse)

        # if layer is set to -2, these are used
        pos_pred_inputs = top_recur
        self.print_once("Setting pos_pred_inputs to: %s" % top_recur.name)
        predicate_inputs = top_recur
        self.print_once("Setting predicate_inputs to: %s" % top_recur.name)

        # Project for Tranformer / residual LSTM input
        if self.n_recur > 0:
          if self.dist_model == "transformer":
            with tf.variable_scope('proj1', reuse=reuse):
              top_recur = self.MLP(top_recur, hidden_size, n_splits=1)
          if self.lstm_residual and self.dist_model == "bilstm":
            with tf.variable_scope('proj1', reuse=reuse):
              top_recur = self.MLP(top_recur, (2 if self.recur_bidir else 1) * self.recur_size, n_splits=1)

        # if layer is set to -1, these are used
        if self.pos_layer == -1:
          pos_pred_inputs = top_recur
          self.print_once("Setting pos_pred_inputs to: %s" % top_recur.name)
        if self.predicate_layer == -1:
          predicate_inputs = top_recur
          self.print_once("Setting predicate_inputs to: %s" % top_recur.name)

        ##### Transformer #######
        if self.dist_model == 'transformer':
          with tf.variable_scope('Transformer', reuse=reuse):
            top_recur = nn.add_timing_signal_1d(top_recur)
            for i in range(self.n_recur):
              with tf.variable_scope('layer%d' % i, reuse=reuse):
                manual_attn = None
                hard_attn = False
                # todo make this into gold_at_train and gold_at_test flags... + scheduled sampling
                if 'parents' in self.multi_layers.keys() and i in self.multi_layers['parents'] and (use_gold_parse or self.full_parse):
                  # if use_gold_parse:
                  #   manual_attn = adj
                  #   # manual_attn = tf.Print(manual_attn, [tf.shape(manual_attn), manual_attn], "gold attn", summarize=100)
                  # if self.full_parse:
                    arc_logits, dep_rel_mlp, head_rel_mlp = get_parse_logits(top_recur)
                  #   # arc_logits = tf.Print(arc_logits, [tf.shape(arc_logits), arc_logits], "arc_logits", summarize=100)
                  #   # if not use_gold_parse:
                  #   #   # compute full parse and set it here
                    manual_attn = tf.cond(use_gold_parse_tensor, lambda: adj, lambda: tf.nn.softmax(arc_logits))
                this_layer_capsule_heads = self.num_capsule_heads if i > 0 else 0
                if 'children' in self.multi_layers.keys() and i in self.multi_layers['children']:
                  this_layer_capsule_heads = 1
                  if use_gold_parse:
                    manual_attn = tf.transpose(adj, [0, 2, 1])
                self.print_once("Layer %d capsule heads: %d" % (i, this_layer_capsule_heads))

                # if use_gold_parse:
                #   if 'parents' in self.multi_layers.keys() and i in self.multi_layers['parents']:
                #     manual_attn = adj
                #   elif 'grandparents' in self.multi_layers.keys() and i in self.multi_layers['grandparents']:
                #     manual_attn = grand_adj
                #   elif 'children' in self.multi_layers.keys() and i in self.multi_layers['children']:
                #     manual_attn = tf.transpose(adj, [0, 2, 1])
                # only at test time
                if moving_params is not None and self.hard_attn:
                  hard_attn = True

                # if 'children' in self.multi_layers.keys() and i in self.multi_layers['children'] and \
                #     self.multi_penalties['children'] != 0.:
                #   this_layer_capsule_heads = 1

                # else:
                top_recur, attn_weights = self.transformer(top_recur, hidden_size, self.num_heads,
                                                           self.attn_dropout, self.relu_dropout, self.prepost_dropout,
                                                           self.relu_hidden_size, self.info_func, self.ff_kernel, reuse,
                                                           this_layer_capsule_heads, manual_attn, hard_attn)
                # head x batch x seq_len x seq_len
                attn_weights_by_layer[i] = tf.transpose(attn_weights, [1, 0, 2, 3])

                if i == self.pos_layer:
                  pos_pred_inputs = top_recur
                  self.print_once("Setting pos_pred_inputs to: %s" % top_recur.name)
                if i == self.predicate_layer:
                  predicate_inputs = top_recur
                  self.print_once("Setting predicate_inputs to: %s" % top_recur.name)
                if i == self.parse_layer:
                  parse_pred_inputs = top_recur
                  self.print_once("Setting parse_pred_inputs to: %s" % top_recur.name)

            # if normalization is done in layer_preprocess, then it should also be done
            # on the output, since the output can grow very large, being the sum of
            # a whole stack of unnormalized layer outputs.
            if self.n_recur > 0:
              top_recur = nn.layer_norm(top_recur, reuse)

        ##### BiLSTM #######
        if self.dist_model == 'bilstm':
          with tf.variable_scope("BiLSTM", reuse=reuse):
            for i in range(self.n_recur):
              with tf.variable_scope('layer%d' % i, reuse=reuse):
                if self.lstm_residual:
                  top_recur_curr, _ = self.RNN(top_recur)
                  top_recur += top_recur_curr
                  # top_recur = nn.layer_norm(top_recur, reuse)
                else:
                  top_recur, _ = self.RNN(top_recur)
            # if self.lstm_residual and self.n_recur > 0:
            #   top_recur = nn.layer_norm(top_recur, reuse)
        if self.num_blocks > 1:
          top_recur = nn.layer_norm(top_recur, reuse)

        if self.pos_layer == self.n_recur - 1:
          pos_pred_inputs = top_recur
          self.print_once("Setting pos_pred_inputs to: %s" % top_recur.name)
        if self.predicate_layer == self.n_recur - 1:
          predicate_inputs = top_recur
          self.print_once("Setting predicate_inputs to: %s" % top_recur.name)
        if self.parse_layer == self.n_recur - 1:
          parse_pred_inputs = top_recur
          self.print_once("Setting parse_pred_inputs to: %s" % top_recur.name)


    ####### 2D CNN ########
    # if self.cnn2d_layers > 0:
    #   with tf.variable_scope('proj2', reuse=reuse):
    #     top_recur_rows, top_recur_cols = self.MLP(top_recur, self.cnn_dim_2d//2, n_splits=2)
    #     # top_recur_rows, top_recur_cols = self.MLP(top_recur, self.cnn_dim // 4, n_splits=2)
    #
    #   top_recur_rows = nn.add_timing_signal_1d(top_recur_rows)
    #   top_recur_cols = nn.add_timing_signal_1d(top_recur_cols)
    #
    #   with tf.variable_scope('2d', reuse=reuse):
    #     # set up input (split -> 2d)
    #     input_shape = tf.shape(embed_inputs)
    #     bucket_size = input_shape[1]
    #     top_recur_rows = tf.tile(tf.expand_dims(top_recur_rows, 1), [1, bucket_size, 1, 1])
    #     top_recur_cols = tf.tile(tf.expand_dims(top_recur_cols, 2), [1, 1, bucket_size, 1])
    #     top_recur_2d = tf.concat([top_recur_cols, top_recur_rows], axis=-1)
    #
    #     # apply num_convs 2d conv layers (residual)
    #     for i in xrange(self.cnn2d_layers):  # todo pass this in
    #       with tf.variable_scope('CNN%d' % i, reuse=reuse):
    #         top_recur_2d += self.CNN(top_recur_2d, kernel, kernel, self.cnn_dim_2d,  # todo pass this in
    #                                 self.recur_keep_prob if i < self.cnn2d_layers - 1 else 1.0,
    #                                 self.info_func if i < self.cnn2d_layers - 1 else tf.identity)
    #         top_recur_2d = nn.layer_norm(top_recur_2d, reuse)
    #
    #     with tf.variable_scope('Arcs', reuse=reuse):
    #       arc_logits = self.MLP(top_recur_2d, 1, n_splits=1)
    #       arc_logits = tf.squeeze(arc_logits, axis=-1)
    #       arc_output = self.output_svd(arc_logits, targets[:, :, 1])
    #       if moving_params is None:
    #         predictions = targets[:, :, 1]
    #       else:
    #         predictions = arc_output['predictions']
    #
    #     # Project each predicted (or gold) edge into head and dep rel representations
    #     with tf.variable_scope('MLP', reuse=reuse):
    #       # flat_labels = tf.reshape(predictions, [-1])
    #       original_shape = tf.shape(arc_logits)
    #       batch_size = original_shape[0]
    #       bucket_size = original_shape[1]
    #       # num_classes = len(vocabs[2])
    #       i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    #       targ = i1 * bucket_size * bucket_size + i2 * bucket_size + predictions
    #       idx = tf.reshape(targ, [-1])
    #       conditioned = tf.gather(tf.reshape(top_recur_2d, [-1, self.cnn_dim_2d]), idx)
    #       conditioned = tf.reshape(conditioned, [batch_size, bucket_size, self.cnn_dim_2d])
    #       dep_rel_mlp, head_rel_mlp = self.MLP(conditioned, self.class_mlp_size + self.attn_mlp_size, n_splits=2)
    # else:

    # if arc_logits already computed, return them. else if arc_loss_penalty != 0, compute them, else dummy
    # arc_logits, dep_rel_mlp, head_rel_mlp = tf.cond(tf.greater(self.arc_loss_penalty, 0.0),
    #                                                 lambda: tf.cond(tf.equal(int(self.full_parse), 1),
    #                                                                 lambda: (arc_logits, dep_rel_mlp, head_rel_mlp),
    #                                                                 lambda: get_parse_logits(parse_pred_inputs)),
    #                                                 lambda: dummy_parse_logits())

    if not self.full_parse and self.role_loss_penalty == 0. and self.predicate_loss_penalty == 0.0:
      arc_logits, dep_rel_mlp, head_rel_mlp = get_parse_logits(parse_pred_inputs)

    arc_output = self.output_svd(arc_logits, targets[:,:,1])
    if moving_params is None:
      predictions = targets[:,:,1]
    else:
      predictions = arc_output['predictions']
    parse_probs = arc_output['probabilities']

    ######## do parse-specific stuff (rels) ########

    def get_parse_rel_logits():
      with tf.variable_scope('Rels', reuse=reuse):
        rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_rel_classes, predictions)
      return rel_logits, rel_logits_cond

    rel_logits, rel_logits_cond = tf.cond(tf.not_equal(self.rel_loss_penalty, 0.0),
                                          lambda: get_parse_rel_logits(),
                                          lambda: (tf.constant(0.), tf.constant(0.)))
    rel_output = self.output(rel_logits, targets[:, :, 2], num_rel_classes)
    rel_output['probabilities'] = tf.cond(tf.not_equal(self.rel_loss_penalty, 0.0),
                                          lambda: self.conditional_probabilities(rel_logits_cond),
                                          lambda: rel_output['probabilities'])

    # def compute_rels_output():
    #   with tf.variable_scope('Rels', reuse=reuse):
    #     rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
    #     rel_output = self.output(rel_logits, targets[:, :, 2])
    #     rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
    #     return rel_output
    # def dummy_compute_rels_output():

    multitask_losses = {}
    multitask_correct = {}
    multitask_loss_sum = 0
    # multitask_parents_preds = arc_logits
    ##### MULTITASK ATTN LOSS ######
    if not self.full_parse:
      for l, attn_weights in attn_weights_by_layer.iteritems():
        # attn_weights is: head x batch x seq_len x seq_len
        # idx into attention heads
        attn_idx = self.num_capsule_heads
        cap_attn_idx = 0
        if 'parents' in self.multi_layers.keys() and l in self.multi_layers['parents']:
          outputs = self.output(attn_weights[attn_idx], multitask_targets['parents'])
          parse_probs = tf.nn.softmax(attn_weights[attn_idx])
          # todo this is a bit of a hack
          attn_idx += 1
          loss = self.multi_penalties['parents'] * outputs['loss']
          multitask_losses['parents%s' % l] = loss
          multitask_correct['parents%s' % l] = outputs['n_correct']
          multitask_loss_sum += loss
        if 'grandparents' in self.multi_layers.keys() and l in self.multi_layers['grandparents']:
          outputs = self.output(attn_weights[attn_idx], multitask_targets['grandparents'])
          attn_idx += 1
          loss = self.multi_penalties['grandparents'] * outputs['loss']
          multitask_losses['grandparents%s' % l] = loss
          multitask_loss_sum += loss
        if 'children' in self.multi_layers.keys() and l in self.multi_layers['children']:
          outputs = self.output_transpose(attn_weights[cap_attn_idx], multitask_targets['children'])
          cap_attn_idx += 1
          loss = self.multi_penalties['children'] * outputs['loss']
          multitask_losses['children%s' % l] = loss
          multitask_loss_sum += loss

    ######## Predicate detection ########
    # predicate_targets = tf.where(tf.greater(targets[:, :, 3], self.predicate_true_start_idx), tf.ones([batch_size, bucket_size], dtype=tf.int32),
    #                            tf.zeros([batch_size, bucket_size], dtype=tf.int32))
    predicate_targets = inputs[:, :, 3]

    def compute_predicates(predicate_input, name):
      with tf.variable_scope(name, reuse=reuse):
        predicate_classifier_mlp = self.MLP(predicate_input, self.predicate_pred_mlp_size, n_splits=1)
        with tf.variable_scope('SRL-Predicates-Classifier', reuse=reuse):
          predicate_classifier = self.MLP(predicate_classifier_mlp, num_pred_classes, n_splits=1)
        output = self.output_predicates(predicate_classifier, predicate_targets, vocabs[4].predicate_true_start_idx)
        return output

    # aux_trigger_loss = tf.constant(0.)
    # if self.train_aux_trigger_layer:
    #   aux_trigger_output = compute_predicates(aux_trigger_inputs, 'SRL-Triggers-Aux', False)
    #   aux_trigger_loss = self.aux_trigger_penalty * aux_trigger_output['loss']

    #print('Start idx: ', vocabs[4].predicate_true_start_idx)
    predicate_targets_binary = tf.where(tf.greater(predicate_targets, vocabs[4].predicate_true_start_idx),
                                     tf.ones_like(predicate_targets), tf.zeros_like(predicate_targets))

    #predicate_targets_binary = tf.Print(predicate_targets_binary, [tf.shape(predicate_targets_binary), predicate_targets_binary, predicate_targets], "predicate targets", summarize=200)

    def dummy_predicate_output():
      return {
        'loss': 0.0,
        'predicate_predictions': predicate_targets_binary,
        'predictions': predicate_targets,
        'logits': 0.0,
        # 'gold_trigger_predictions': tf.transpose(predictions, [0, 2, 1]),
        'count': 0.,
        'correct': 0.,
        'targets': 0,
      }

    predicate_output = tf.cond(tf.greater(self.predicate_loss_penalty, 0.0),
                               lambda: compute_predicates(predicate_inputs, 'SRL-Predicates'),
                               dummy_predicate_output)

    if moving_params is None or self.add_predicates_to_input or self.predicate_loss_penalty == 0.0:
      # gold
      predicate_predictions = predicate_targets_binary
    else:
      # predicted
      predicate_predictions = predicate_output['predicate_predictions']

    #predicate_predictions = tf.Print(predicate_predictions, [predicate_targets], "predicate_targets", summarize=50)
    #predicate_predictions = tf.Print(predicate_predictions, [predicate_predictions], "predicate_predictions", summarize=50)


    ######## POS tags ########
    def compute_pos(pos_input, pos_target):
        with tf.variable_scope('POS-Classifier', reuse=reuse):
          pos_classifier = self.MLP(pos_input, num_pos_classes, n_splits=1)
        output = self.output(pos_classifier, pos_target)
        return output

    pos_target = targets[:,:,0]
    pos_loss = tf.constant(0.)
    pos_correct = tf.constant(0.)
    pos_preds = pos_target
    if self.train_pos:
      pos_output = compute_pos(pos_pred_inputs, pos_target)
      pos_loss = self.pos_penalty * pos_output['loss']
      pos_correct = pos_output['n_correct']
      pos_preds = pos_output['predictions']
    elif self.joint_pos_predicates:
      pos_preds = tf.squeeze(tf.nn.embedding_lookup(preds_to_pos_map, predicate_output['predictions']), -1)
      pos_correct = tf.reduce_sum(tf.cast(tf.equal(pos_preds, pos_target), tf.float32) * tf.squeeze(self.tokens_to_keep3D, -1))
    elif self.add_pos_to_input:
      pos_correct = tf.reduce_sum(tf.cast(tf.equal(inputs[:,:,2], pos_target), tf.float32) * tf.squeeze(self.tokens_to_keep3D, -1))
      pos_preds = inputs[:,:,2]

    ######## do SRL-specific stuff (rels) ########
    def compute_srl(srl_target, vn_embeddings, num_classes):
      with tf.variable_scope('SRL-MLP', reuse=reuse):
        # project to get predicate and role representations?
        predicate_role_mlp = self.MLP(top_recur, self.predicate_mlp_size + self.role_mlp_size, n_splits=1)
        predicate_mlp, role_mlp = predicate_role_mlp[:,:,:self.predicate_mlp_size], predicate_role_mlp[:, :, self.predicate_mlp_size:]

      with tf.variable_scope('SRL-Arcs', reuse=reuse):
        # gather just the triggers
        # predicate_predictions: batch x seq_len
        # gathered_predicates: num_triggers_in_batch x 1 x self.trigger_mlp_size
        # role mlp: batch x seq_len x self.role_mlp_size
        # gathered roles: need a (bucket_size x self.role_mlp_size) role representation for each trigger,
        # i.e. a (num_triggers_in_batch x bucket_size x self.role_mlp_size) tensor
        predicate_gather_indices = tf.where(tf.equal(predicate_predictions, 1))
        # predicate_gather_indices = tf.Print(predicate_gather_indices, [predicate_predictions, tf.shape(predicate_gather_indices), tf.shape(predicate_predictions)], "predicate gather shape", summarize=200)
        gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
        tiled_roles = tf.reshape(tf.tile(role_mlp, [1, bucket_size, 1]), [batch_size, bucket_size, bucket_size, self.role_mlp_size])
        gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
        if combine_with_vn:
          derived_roles = tf.add(gathered_roles, vn_embeddings)
        else:
          derived_roles = gathered_roles

        # now multiply them together to get (num_triggers_in_batch x bucket_size x num_srl_classes) tensor of scores
        srl_logits = self.bilinear_classifier_nary(gathered_predicates, derived_roles, num_classes)
        srl_logits_transpose = tf.transpose(srl_logits, [0, 2, 1])
        srl_output = self.output_srl_gather(srl_logits_transpose, srl_target, predicate_predictions, transition_params if self.viterbi_train else None, None)
        return srl_output, gathered_predicates, gathered_roles

    def compute_vn(vn_target, num_classes):
      with tf.variable_scope('VN-MLP', reuse=reuse):
        predicate_role_mlp = self.MLP(top_recur, self.predicate_mlp_size + self.role_mlp_size, n_splits=1)
        predicate_mlp, role_mlp = predicate_role_mlp[:,:,:self.predicate_mlp_size], predicate_role_mlp[:, :, self.predicate_mlp_size:]

      with tf.variable_scope('VN-Arcs', reuse=reuse):
        # gather just the triggers
        # predicate_predictions: batch x seq_len
        # gathered_predicates: num_triggers_in_batch x 1 x self.trigger_mlp_size
        # role mlp: batch x seq_len x self.role_mlp_size
        # gathered roles: need a (bucket_size x self.role_mlp_size) role representation for each trigger,
        # i.e. a (num_triggers_in_batch x bucket_size x self.role_mlp_size) tensor
        predicate_gather_indices = tf.where(tf.equal(predicate_predictions, 1))
        # predicate_gather_indices = tf.Print(predicate_gather_indices, [predicate_predictions, tf.shape(predicate_gather_indices), tf.shape(predicate_predictions)], "predicate gather shape", summarize=200)
        gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
        tiled_roles = tf.reshape(tf.tile(role_mlp, [1, bucket_size, 1]), [batch_size, bucket_size, bucket_size, self.role_mlp_size])
        gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)

        # now multiply them together to get (num_triggers_in_batch x bucket_size x num_srl_classes) tensor of scores
        vn_logits = self.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_classes)
        logits_transpose = tf.transpose(vn_logits, [0, 2, 1])
        # logits_transpose = tf.Print(logits_transpose, [tf.shape(logits_transpose)], "vn_logits_transpose")
        vn_output = self.output_srl_gather(logits_transpose, vn_target, predicate_predictions, None, annotated_3D, vn_nolabel_idx)

        return vn_output

    def compute_vn_rep(vn_target, vn_logits):
      preds_in_batch = tf.reduce_sum(predicate_targets_binary)
      bucket_size = tf.shape(vn_target)[1]
      #vn_logits = tf.Print(vn_logits, [tf.shape(vn_logits), tf.shape(vn_target), preds_in_batch], "logits and target ")
      logits_shape = tf.shape(vn_logits)
      num_classes = num_vn_classes
      if self.gold_train_vn and (self.gold_test_vn or dataset.name == 'Trainset'):
        print('using gold vn')
        trigger_counts = tf.reduce_sum(predicate_predictions, -1)
        vn_targets_indices = tf.where(tf.sequence_mask(tf.reshape(trigger_counts, [-1])))
        vn_targets_gathered = tf.gather_nd(tf.transpose(vn_target, [0, 2, 1]), vn_targets_indices)

        vn_targets_one_hot = tf.one_hot(vn_targets_gathered, depth=num_classes, axis=-1)

        vn_scores = tf.reshape(vn_targets_one_hot, [preds_in_batch * bucket_size, num_classes])
      else:
        print('using predicted vn')
        vn_scores = tf.reshape(tf.nn.softmax(vn_logits, axis=-1), [preds_in_batch * bucket_size, num_classes])

      vn_embeddings = vocabs[6].embedding_lookup(tf.range(num_classes), moving_params=self.moving_params)

      weighted_vn_embeddings = tf.reshape(tf.matmul(vn_scores, vn_embeddings), [preds_in_batch, bucket_size, 200])

      return weighted_vn_embeddings

    def compute_srl_simple(srl_target):
      with tf.variable_scope('SRL-MLP', reuse=reuse):
        # srl_logits are batch x seq_len x num_classes
        srl_logits = self.MLP(top_recur, num_srl_classes, n_splits=1)
        # srl_target is targets[:, :, 3:]: batch x seq_len x targets
        output = self.output(srl_logits, srl_target)
        srl_output = {f: output[f] for f in ['loss', 'probabilities', 'predictions', 'correct', 'count']}
        srl_output['logits'] = srl_logits
        srl_output['transition_params'] = tf.constant(0.)
        srl_output['correct'] = output['n_correct']
        return srl_output

    if self.role_loss_penalty == 0:
      num_triggers = tf.reduce_sum(tf.cast(tf.where(tf.equal(predicate_targets_binary, 1)), tf.int32))
      srl_output = {
        'loss': tf.constant(0.),
        'probabilities':  tf.zeros([num_triggers, bucket_size, num_srl_classes]),
        'predictions': tf.reshape(tf.transpose(srl_targets, [0, 2, 1]), [-1, bucket_size]), # tf.zeros([num_triggers, bucket_size]),
        'logits':  tf.zeros([num_triggers, bucket_size, num_srl_classes]),
        'correct':  tf.constant(0.),
        'count':  tf.constant(1.)
      }
      vn_output, vn_embeddings = compute_vn(vn_targets, num_vn_classes)
    elif self.srl_simple_tagging:
      srl_output = compute_srl_simple(srl_targets)
    else:
      if predict_vn:
        vn_output = compute_vn(vn_targets, num_vn_classes)
      else:
        num_triggers = tf.reduce_sum(predicate_targets_binary)
        bucket_size = tf.shape(vn_targets)[1]
        vn_output = {
          'loss': tf.constant(0.),
          'probabilities': tf.zeros([num_triggers, bucket_size, num_vn_classes]),
          'predictions': tf.reshape(tf.transpose(vn_targets, [0, 2, 1]), [-1, bucket_size]),
        # tf.zeros([num_triggers, bucket_size]),
          'logits': tf.zeros([num_triggers, bucket_size, num_vn_classes]),
          'correct': tf.constant(0.),
          'count': tf.constant(1.),
          'targets': tf.constant(0.),
          'preds_to_keep': tf.ones([num_triggers,1])
        }

      vn_embeddings = compute_vn_rep(vn_targets, vn_output['logits'])

      srl_output, srl_pred_reps, srl_role_reps = compute_srl(srl_targets, vn_embeddings, num_srl_classes)

    predicate_loss = self.predicate_loss_penalty * predicate_output['loss']
    srl_loss = self.role_loss_penalty * srl_output['loss']
    vn_loss = self.vn_loss_penalty * vn_output['loss']
    arc_loss = self.arc_loss_penalty * arc_output['loss']
    rel_loss = self.rel_loss_penalty * rel_output['loss']

    # if this is a parse update, then actual parse loss equal to sum of rel loss and arc loss
    # actual_parse_loss = tf.cond(tf.equal(int(self.full_parse), 1), lambda: tf.add(rel_loss, arc_loss), lambda: tf.constant(0.))
    # actual_parse_loss = tf.cond(do_parse_update, lambda: tf.add(rel_loss, arc_loss), lambda: tf.constant(0.))
    parse_combined_loss = rel_loss + arc_loss

    # if this is a parse update and the parse proportion is not one, then no srl update. otherwise,
    # srl update equal to sum of srl_loss, predicate_loss
    srl_combined_loss = srl_loss + predicate_loss
    # actual_srl_loss = tf.cond(tf.logical_and(do_parse_update, tf.not_equal(self.parse_update_proportion, 1.0)), lambda: tf.constant(0.), lambda: srl_combined_loss)

    output = {}

    output['multitask_losses'] = multitask_losses

    output['probabilities'] = tf.tuple([parse_probs,
                                        rel_output['probabilities']])
    output['predictions'] = tf.stack([arc_output['predictions'],
                                      rel_output['predictions']])
    output['correct'] = arc_output['correct'] * rel_output['correct']
    output['tokens'] = arc_output['tokens']
    output['n_correct'] = tf.reduce_sum(output['correct'])
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = output['n_correct'] / output['n_tokens']

    if predict_vn:
      output['loss'] = srl_combined_loss + parse_combined_loss + multitask_loss_sum + pos_loss + vn_loss
    else:
      output['loss'] = srl_combined_loss + parse_combined_loss + multitask_loss_sum + pos_loss
    # output['loss'] = srl_loss + predicate_loss + actual_parse_loss
    # output['loss'] = actual_srl_loss + arc_loss + rel_loss

    if self.word_l2_reg > 0:
      output['loss'] += word_loss

    output['embed'] = embed_inputs
    output['recur'] = top_recur
    # output['dep_arc'] = dep_arc_mlp
    # output['head_dep'] = head_arc_mlp
    output['dep_rel'] = dep_rel_mlp
    output['head_rel'] = head_rel_mlp
    output['arc_logits'] = arc_logits
    output['rel_logits'] = rel_logits

    output['rel_loss'] = rel_loss # rel_output['loss']
    output['log_loss'] = arc_loss # arc_output['log_loss']
    output['2cycle_loss'] = arc_output['2cycle_loss']
    output['roots_loss'] = arc_output['roots_loss']
    output['svd_loss'] = arc_output['svd_loss']
    output['n_cycles'] = arc_output['n_cycles']
    output['len_2_cycles'] = arc_output['len_2_cycles']

    output['srl_loss'] = srl_loss
    output['vn_loss'] = vn_loss
    output['srl_preds'] = srl_output['predictions']
    output['srl_probs'] = srl_output['probabilities']
    output['srl_logits'] = srl_output['logits']
    output['srl_correct'] = srl_output['correct']
    output['srl_count'] = srl_output['count']
    output['vn_preds'] = vn_output['predictions']
    output['vn_probs'] = vn_output['probabilities']
    output['vn_logits'] = vn_output['logits']
    output['vn_correct'] = vn_output['correct']
    output['vn_count'] = vn_output['count']
    output['vn_targets'] = vn_output['targets']
    output['transition_params'] = transition_params if transition_params is not None else tf.constant(bilou_constraints)
    output['srl_predicates'] = predicate_predictions
    output['srl_predicate_targets'] = predicate_targets_binary
    output['predicate_loss'] = predicate_loss
    output['predicate_count'] = predicate_output['count']
    output['predicate_correct'] = predicate_output['correct']
    output['predicate_preds'] = predicate_output['predictions']
    output['preds_to_keep'] = vn_output['preds_to_keep']

    output['sample_prob'] = sample_prob

    output['pos_loss'] = pos_loss
    output['pos_correct'] = pos_correct
    output['pos_preds'] = pos_preds

    # transpose and softmax attn weights
    attn_weights_by_layer_softmaxed = {k: tf.transpose(tf.nn.softmax(v), [1, 0, 2, 3]) for k, v in
                                       attn_weights_by_layer.iteritems()}

    output['attn_weights'] = attn_weights_by_layer_softmaxed
    output['attn_correct'] = multitask_correct

    return output
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep, n_cycles=-1, len_2_cycles=-1):
    """"""
    start_time = time.time()
    parse_preds, roots_lt, roots_gt = self.parse_argmax(parse_probs, tokens_to_keep, n_cycles, len_2_cycles)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    total_time = time.time() - start_time
    return parse_preds, rel_preds, total_time, roots_lt, roots_gt
