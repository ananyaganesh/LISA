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

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models import NN

#***************************************************************
class BaseParser(NN):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    raise NotImplementedError
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep, n_cycles=-1, len_2_cycles=-1):
    """"""
    
    raise NotImplementedError
  
  #=============================================================
  def sanity_check(self, inputs, targets, predictions, vocabs, fileobject, feed_dict={}):
    """"""
    
    for tokens, golds, parse_preds, rel_preds in zip(inputs, targets, predictions[0], predictions[1]):
      # print("tokens", tokens)
      # print("golds", golds)
      for l, (token, gold, parse, rel) in enumerate(zip(tokens, golds, parse_preds, rel_preds)):
        if token[0] > 0:
          word = vocabs[0][token[0]]
          glove = vocabs[0].get_embed(token[1])
          tag = vocabs[1][token[2]]
          gold_tag = vocabs[1][gold[0]]
          pred_parse = parse
          pred_rel = vocabs[2][rel]
          gold_parse = gold[1]
          gold_rel = vocabs[2][gold[2]]
          fileobject.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t%d\t%s\n' % (l, word, glove, tag, gold_tag, pred_parse, pred_rel, gold_parse, gold_rel))
      fileobject.write('\n')
    return
  
  #=============================================================
  def validate(self, mb_inputs, mb_targets, annotated, mb_probs, n_cycles, len_2_cycles, srl_preds, srl_logits, srl_triggers, srl_trigger_targets, pos_preds, vn_preds, vn_logits, vn_targets, preds_to_keep, transition_params=None):
    """"""
    
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    total_time = 0.0
    roots_lt_total = 0.
    roots_gt_total = 0.
    cycles_2_total = 0.
    cycles_n_total = 0.
    non_trees_total = 0.
    non_tree_preds = []
    # srl_triggers = np.transpose(srl_triggers)
    # np.set_printoptions(threshold=np.nan)
    # print("triggers", srl_triggers)
    if np.all(n_cycles == -1):
        n_cycles = len_2_cycles = [-1] * len(mb_inputs)

    # for each batch element (sequence)
    # need to index into srl_preds, srl_logits

    # print("mb probs", mb_probs)
    # print("mb_parse_probs", mb_parse_probs)

    # print("srl_preds", srl_preds.shape, srl_preds)
    # print("srl_trigger", srl_triggers.shape, srl_triggers)
    srl_pred_idx = 0
    n_tokens = 0.
    for inputs, targets, annot, parse_probs, rel_probs, n_cycle, len_2_cycle, srl_trigger, srl_trigger_target, pos_pred in zip(mb_inputs, mb_targets, annotated, mb_parse_probs, mb_rel_probs, n_cycles, len_2_cycles, srl_triggers, srl_trigger_targets, pos_preds):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      n_tokens += length
      parse_preds, rel_preds, argmax_time, roots_lt, roots_gt = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep, n_cycle, len_2_cycle)
      total_time += argmax_time
      roots_lt_total += roots_lt
      roots_gt_total += roots_gt
      cycles_2_total += int(len_2_cycle)
      cycles_n_total += int(n_cycle)
      if roots_lt or roots_gt or len_2_cycle or n_cycle:
        non_trees_total += 1.
        non_tree_preds.append((parse_probs, targets, length, int(len_2_cycle), int(n_cycle)))
      # targets has 3 non-srl things, then srls, variable length
      non_srl_targets_len = 3
      tokens = np.arange(length)
      # print(srl_trigger)
      # print(srl_trigger_target)
      pred_trigger_indices = np.where(srl_trigger[tokens] == 1)[0]
      gold_trigger_indices = np.where(srl_trigger_target[tokens] == 1)[0]
      num_gold_srls = len(gold_trigger_indices)
      num_pred_srls = len(pred_trigger_indices)

      # num_triggers x seq_len
      # print(srl_preds)
      srl_pred = srl_preds[srl_pred_idx:srl_pred_idx+num_pred_srls, tokens]
      vn_pred = vn_preds[srl_pred_idx:srl_pred_idx+num_pred_srls, tokens]
      # vn_targ = vn_targets[srl_pred_idx:srl_pred_idx+num_pred_srls, tokens]

      # print("vn pred keep", vn_pred_keep)

      vn_pred_keep = np.squeeze(preds_to_keep[srl_pred_idx:srl_pred_idx+num_pred_srls], -1)

      # vn_pred_keep = preds_to_keep[srl_pred_idx:srl_pred_idx+num_pred_srls]

      num_vns = int(np.sum(vn_pred_keep))
      vn_pred = vn_pred[np.where(vn_pred_keep == 1)]

      # print("vn pred shape after", vn_pred.shape)

      #print('Annotation: ', annot, 'Verbnet: ', vn_pred)

      #print("srl pred", len(vn_pred), vn_pred)
      #print('srl logits shape: ', srl_logits.shape, 'vn logits shape: ', vn_logits.shape)

      if transition_params is not None and num_pred_srls > 0:
        srl_unary_scores = srl_logits[srl_pred_idx:srl_pred_idx+num_pred_srls, tokens]
        vn_unary_scores = vn_logits[srl_pred_idx:srl_pred_idx+num_pred_srls, tokens]
        # print("unary scores shape", srl_unary_scores.shape)
        for pred_idx, single_pred_unary_scores in enumerate(srl_unary_scores):
          viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(single_pred_unary_scores, transition_params)
          # print("viterbi seq", len(viterbi_sequence), viterbi_sequence)
          srl_pred[pred_idx] = viterbi_sequence

      srl_pred_idx += num_pred_srls

      # print("s_pred shape", srl_pred.shape)
      # print("num pred srls", num_pred_srls)
      # if num_pred_srls > 0:
      #   np.set_printoptions(threshold=np.nan)
      #   print("srl_trigger", srl_trigger)
      #   print("shape, tokens", srl_pred.shape, length)
      #   print("srl pred", srl_pred)
      #   print("srl pred where", srl_pred[:, pred_trigger_indices])
      #   print("trigger indices", pred_trigger_indices)

        # print("gold where", np.where(targets[tokens, non_srl_targets_len:] == trigger_idx))
        # print("gold", targets[tokens, non_srl_targets_len:])
        # print("gold where", targets[tokens, non_srl_targets_len:num_gold_srls+non_srl_targets_len])

      # print("num srls", num_srls)
      # print("where", np.where(targets[tokens, non_srl_targets_len:] == trigger_idx))
      # print("len", len(np.where(targets[tokens, non_srl_targets_len:] == trigger_idx)))


      # num_srls = targets.shape[-1]-non_srl_targets_len
      # sent will contain 7 things non-srl, including one thing from targets
      sent = -np.ones((length, 2*num_pred_srls+3*num_gold_srls+16+num_vns), dtype=int)

      # print("sent shape", sent.shape)

      # print("srl targets", targets[tokens, 3:])
      # print("srl triggers", np.sum(np.where(targets[tokens, 3:] == trigger_idx)))

      #print("inputs", inputs.shape)
      # print("srl pred shape", srl_pred.shape)
      # print("srl pred", srl_pred)
      # print("srl pred[tokens]", srl_pred[tokens])
      # print("num_srls", num_srls)
      #print("targets shape", targets.shape)
      # print("targets", targets)
      #print("tokens", tokens.shape)
      #print('Annotation shape: ', annot.shape)

      sent[:,0] = tokens # 1 = index
      sent[:,1:7] = inputs[tokens,:] # 2,3,4,5,6,7 inputs[0, 1, 2, 3, 4, 5] = word, word, auto_tag, predicate t/f, domain, sent id
      sent[:,7] = targets[tokens, 0] # 5 targets[0] = gold_tag
      sent[:,8] = parse_preds[tokens] # 6 = pred parse head
      sent[:,9] = rel_preds[tokens] # 7 = pred parse label
      sent[:,10] = targets[tokens, 1] # 8 = gold parse head
      sent[:,11] = targets[tokens, 2] # 9 = gold parse label
      sent[:,12] = pos_pred[tokens] # 10 = predicted pos label
      sent[:,13] = num_gold_srls # 11 = num gold predicates in sent
      sent[:,14] = num_pred_srls  # 12 = num predicted predicates in sent
      sent[:,15] = annot[0]
      sent[:,16:16+num_pred_srls] = pred_trigger_indices # indices of predicted predicates
      # save trigger indices
      sent[:,16+num_pred_srls:16+num_gold_srls+num_pred_srls] = targets[tokens, non_srl_targets_len:num_gold_srls+non_srl_targets_len] # gold srl tags
      # print("trigger tokens", srl_trigger[tokens])
      # print("indices", np.where(srl_trigger[tokens] == 1)[0])
      # print("srl_pred", srl_pred)
      # print("srl_pred where", srl_pred[:,np.where(srl_trigger[tokens] == 1)[0]])
      s_pred = np.transpose(srl_pred)
      vn_pred_trans = np.transpose(vn_pred)
      # print("srl_pred", srl_pred.shape, srl_pred)
      # print("pred_trigger_indices", pred_trigger_indices)
      #print("s_pred", s_pred.shape, s_pred)
      #print("v_pred", vn_pred_trans.shape, vn_pred_trans)

      if len(s_pred.shape) == 1:
        s_pred = np.expand_dims(s_pred, -1)
      if len(vn_pred_trans.shape) == 1:
        vn_pred_trans = np.expand_dims(vn_pred_trans, -1)
      sent[:,16+num_pred_srls+num_gold_srls:16+2*num_pred_srls+num_gold_srls] = s_pred

      # keep track of vn predicates (binary for each gold pb predicate)
      sent[0, 16+num_gold_srls+2*num_pred_srls:16+2*num_gold_srls+2*num_pred_srls] = vn_pred_keep # vn predicates

      # gold pb predicate indices
      sent[:, 16+2*num_gold_srls+2*num_pred_srls:16+3*num_gold_srls+2*num_pred_srls] = gold_trigger_indices # pb gold predicates

      if num_vns > 0:
        # print("vn pred trans shape", vn_pred_trans.shape)
        sent[:, 16+2*num_pred_srls+3*num_gold_srls:] = vn_pred_trans
      #print(num_pred_srls, num_gold_srls, s_pred.shape[1], sent.shape[1])
      sents.append(sent)
    return sents, total_time, roots_lt_total, roots_gt_total, cycles_2_total, cycles_n_total, non_trees_total, non_tree_preds, n_tokens
  
  #=============================================================
  @staticmethod
  def evaluate(filename, punct=NN.PUNCT):
    """"""
    
    correct = {'UAS': [], 'LAS': []}
    with open(filename) as f:
      for line in f:
        line = line.strip().split('\t')
        if len(line) == 10 and line[4] not in punct:
          correct['UAS'].append(0)
          correct['LAS'].append(0)
          if line[6] == line[8]:
            correct['UAS'][-1] = 1
            if line[7] == line[9]:
              correct['LAS'][-1] = 1
    correct = {k:np.array(v) for k, v in correct.iteritems()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct

  # =============================================================
  @staticmethod
  def evaluate_by_len(filename, punct=NN.PUNCT):
    """"""
    # want UAS broken down by: sentence length, dep distance dep label
    # want LAS broken down by: dep label
    correct = {'UAS': [], 'LAS': []}
    correct_by_sent_len = {}
    correct_by_dep_len = {}
    correct_by_dep = {}
    uas_by_sent_len = {}
    uas_by_dep_len = {}
    las_by_dep = {}
    curr_sent_len = 0
    curr_sent_correct = 0
    curr_sent_pred = 0
    with open(filename) as f:
      for line in f:
        line = line.strip().split('\t')
        if len(line) == 10 and line[4] not in punct:
          correct['UAS'].append(0)
          correct['LAS'].append(0)
          if line[6] == line[8]:
            correct['UAS'][-1] = 1
            if line[7] == line[9]:
              correct['LAS'][-1] = 1
        # elif len(line) != 10:
        #   # update all the counts by sentence

    correct = {k: np.array(v) for k, v in correct.iteritems()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct

  #=============================================================
  @property
  def input_idxs(self):
    # word, word, auto_tag, predicate t/f, domain, sent id
    return (0, 1, 2, 3, 4, 5)
  @property
  def target_idxs(self):
    # need to add target indices here?
    # up to max len?
    # gold tag, gold parse head, gold parse rel
    return (7, 8, 9)
