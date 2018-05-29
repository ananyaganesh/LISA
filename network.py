#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import os
import sys
import time
import pickle as pkl

import numpy as np
import tensorflow as tf
print(tf.__version__)
import keras
from keras.models import Sequential, Model

from lib import models
from lib import optimizers
from lib import rnn_cells

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset
import contextlib
from subprocess import check_output, CalledProcessError
import operator

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

#***************************************************************
class Network(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, model, *args, **kwargs):
    """"""
    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')
    
    kwargs['name'] = kwargs.pop('name', model.__name__)
    super(Network, self).__init__(*args, **kwargs)
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)
    with open(os.path.join(self.save_dir, 'config.cfg'), 'w') as f:
      self._config.write(f)

    # objectives = ['pos_loss', 'trigger_loss', 'actual_parse_loss', 'srl_loss', 'multitask_loss_sum']
    # self._global_steps = {o: tf.Variable(0., trainable=False) for o in objectives}
    self._global_step = tf.Variable(0., trainable=False)
    self._global_epoch = tf.Variable(0., trainable=False)

    # todo what is this??
    # self._model = model(self._config, global_step=self.global_step)
    self._model = model(self._config)
    
    #kmodel = Model.from_config(self._config)
    #kmodel.summary()

    self._vocabs = []

    if self.conll:
      vocab_files = [(self.word_file, 1, 'Words'),
                     (self.tag_file, [3, 4], 'Tags'),
                     (self.rel_file, 7, 'Rels')]
    elif self.conll2012:
      vocab_files = [(self.word_file, 3, 'Words'),
                     (self.tag_file, [5, 4], 'Tags'), # auto, gold
                     (self.rel_file, 7, 'Rels'),
                     (self.srl_file, range(14, 50), 'SRLs'),
                     (self.trig_file, [10, 4] if self.joint_pos_predicates else 10, 'Trigs'),
                     (self.domain_file, 0, 'Domains')]

    print("Loading vocabs")
    sys.stdout.flush()
    for i, (vocab_file, index, name) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    use_pretrained=(not i))
      self._vocabs.append(vocab)

    print("Predicates vocab: ")
    for l, i in sorted(self._vocabs[4].iteritems(), key=operator.itemgetter(1)):
      print("%s: %d" % (l, i))
    print("predicate_true_start_idx", self._vocabs[4].predicate_true_start_idx)

    print("Loading data")
    sys.stdout.flush()
    self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')

    self._ops = self._gen_ops()
    self._save_vars = filter(lambda x: u'Pretrained' not in x.name, tf.global_variables())
    self.history = {
      'train_loss': [],
      'train_accuracy': [],
      'valid_loss': [],
      'valid_accuracy': [],
      'test_acuracy': 0
    }
    return
  
  #=============================================================
  def train_minibatches(self):
    """"""
    
    return self._trainset.get_minibatches(self.train_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs)
  
  #=============================================================
  def valid_minibatches(self):
    """"""
    
    return self._validset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)
  
  #=============================================================
  def test_minibatches(self):
    """"""
    
    return self._testset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)
  
  #=============================================================
  # assumes the sess has already been initialized
  def train(self, sess, profile):
    """"""
    print("Training")
    training_start_time = time.time()
    sys.stdout.flush()
    save_path = os.path.join(self.save_dir, self.name.lower() + '-pretrained')
    saver = tf.train.Saver(self.save_vars, max_to_keep=1, save_relative_paths=True)
    
    n_bkts = self.n_bkts
    train_iters = self.train_iters
    print_every = self.print_every
    validate_every = self.validate_every
    save_every = self.save_every
    current_best = 0.0
    try:
      train_time = 0
      train_loss = 0
      train_log_loss = 0
      train_roots_loss = 0
      train_cycle2_loss = 0
      train_svd_loss = 0
      train_rel_loss = 0
      train_srl_loss = 0
      train_mul_loss = {}
      train_trigger_loss = 0
      train_pos_loss = 0
      n_train_sents = 0
      n_train_correct = 0
      n_train_tokens = 0
      n_train_iters = 0
      n_train_srl_correct = 0
      n_train_srl_count = 0
      n_train_trigger_count = 0
      n_train_trigger_correct = 0
      total_train_iters = 0
      valid_time = 0
      valid_loss = 0
      valid_accuracy = 0
      print('Total train iters: ', total_train_iters)
      print('Train iters: ', train_iters)
      avg_train_time = 0
      avg_val_time = 0
      div_count = 0	
      while total_train_iters < train_iters:
        for j, (feed_dict, _) in enumerate(self.train_minibatches()):
          # train_inputs = feed_dict[self._trainset.inputs]
          train_targets = feed_dict[self._trainset.targets]
          start_time = time.time()

          if profile:
            pctx.trace_next_step()
            # Dump the profile to '/tmp/train_dir' after the step.
            pctx.dump_next_step()

          _, loss, n_correct, n_tokens, roots_loss, cycle2_loss, svd_loss, log_loss, rel_loss, srl_loss, srl_correct, srl_count, trigger_loss, trigger_count, trigger_correct, pos_loss, pos_correct, multitask_losses, lr = sess.run(self.ops['train_op_srl'], feed_dict=feed_dict)
          total_train_iters += 1
          train_time += time.time() - start_time
          train_loss += loss
          train_log_loss += log_loss
          train_roots_loss += roots_loss
          train_cycle2_loss += cycle2_loss
          train_svd_loss += svd_loss
          train_rel_loss += rel_loss
          train_srl_loss += srl_loss
          train_pos_loss += pos_loss
          train_trigger_loss += trigger_loss
          n_train_trigger_count += trigger_count
          n_train_trigger_correct += trigger_correct

          for n, l in multitask_losses.iteritems():
            if n not in train_mul_loss.keys():
              train_mul_loss[n] = 0.
            train_mul_loss[n] += l

          n_train_sents += len(train_targets)
          n_train_correct += n_correct
          n_train_tokens += n_tokens
          n_train_srl_correct += srl_correct
          n_train_srl_count += srl_count
          n_train_iters += 1
          total_train_iters += 1
          self.history['train_loss'].append(loss)
          self.history['train_accuracy'].append(100 * n_correct / n_tokens)
          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            valid_time = 0
            valid_loss = 0
            n_valid_sents = 0
            n_valid_correct = 0
            n_valid_tokens = 0
            with open(os.path.join(self.save_dir, 'sanitycheck.txt'), 'w') as f:
              for k, (feed_dict, _) in enumerate(self.valid_minibatches()):
                inputs = feed_dict[self._validset.inputs]
                targets = feed_dict[self._validset.targets]
                start_time = time.time()
                loss, n_correct, n_tokens, predictions = sess.run(self.ops['valid_op'], feed_dict=feed_dict)
                valid_time += time.time() - start_time
                valid_loss += loss
                n_valid_sents += len(targets)
                n_valid_correct += n_correct
                n_valid_tokens += n_tokens
                self.model.sanity_check(inputs, targets, predictions, self._vocabs, f, feed_dict=feed_dict)
            valid_loss /= k+1
            valid_accuracy = 100 * n_valid_correct / n_valid_tokens
            valid_time = n_valid_sents / valid_time
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_accuracy'].append(valid_accuracy)
          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_log_loss /= n_train_iters
            train_roots_loss /= n_train_iters
            train_cycle2_loss /= n_train_iters
            train_svd_loss /= n_train_iters
            train_rel_loss /= n_train_iters
            train_srl_loss /= n_train_iters
            train_trigger_loss /= n_train_iters
            train_pos_loss /= n_train_iters
            train_accuracy = 100 * n_train_correct / n_train_tokens
            # train_srl_accuracy = 100 * n_train_srl_correct / n_train_srl_count
            # train_trigger_accuracy = 100 * n_train_trigger_correct / n_train_trigger_count
            train_time = n_train_sents / train_time
            print('%6d) Train loss: %.4f    Train acc: %5.2f%%    Train rate: %6.1f sents/sec    Learning rate: %f\n'
                  '\tValid loss: %.4f    Valid acc: %5.2f%%    Valid rate: %6.1f sents/sec' %
                  (total_train_iters, train_loss, train_accuracy, train_time, lr, valid_loss, valid_accuracy, valid_time))
            print('\tlog loss: %f\trel loss: %f\tsrl loss: %f\ttrig loss: %f\tpos loss: %f' % (train_log_loss, train_rel_loss, train_srl_loss, train_trigger_loss, train_pos_loss))
            multitask_losses_str = ''
            for n, l in train_mul_loss.iteritems():
              train_mul_loss[n] = l/n_train_iters
              multitask_losses_str += '\t%s loss: %f' % (n, train_mul_loss[n])
            print(multitask_losses_str)
            sys.stdout.flush()

	    avg_train_time += train_time
	    avg_val_time += valid_time
	    div_count += 1
            train_time = 0
            train_loss = 0
            n_train_sents = 0
            n_train_correct = 0
            n_train_tokens = 0
            n_train_iters = 0
            train_log_loss = 0
            train_roots_loss = 0
            train_cycle2_loss = 0
            train_rel_loss = 0
            train_trigger_loss = 0
            train_srl_loss = 0
            n_train_srl_correct = 0
            n_train_srl_count = 0
            n_train_trigger_correct = 0
            n_train_trigger_count = 0
          if save_every and (total_train_iters % save_every == 0):
            elapsed_time_str = time.strftime("%d:%H:%M:%S", time.gmtime(time.time()-training_start_time))
            print("Elapsed time: %s" % elapsed_time_str)
            with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
              pkl.dump(self.history, f)
            # only look at non-viterbi decoding if we didn't train w/ crf
            current_score = 0.
            if not self.viterbi_train:
              correct = self.test(sess, validate=True)
              current_score = correct[self.eval_criterion]
            if self.viterbi_decode or self.viterbi_train:
              correct = self.test(sess, viterbi=True, validate=True)
              current_score = np.max([correct[self.eval_criterion], current_score])
            # las = np.mean(correct["LAS"]) * 100
            # uas = np.mean(correct["UAS"]) * 100
            # print('UAS: %.2f    LAS: %.2f' % (uas, las))
            if self.save and current_score > current_best:
              current_best = current_score
              print("Writing model to %s" % (os.path.join(self.save_dir, self.name.lower() + '-trained')))
              saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                         latest_filename=self.name.lower(),
                         global_step=self.global_epoch,
                         write_meta_graph=False)
              if self.eval_parse:
                with open(os.path.join(self.save_dir, "parse_results.txt"), 'w') as parse_results_f:
                  print(correct['parse_eval'], file=parse_results_f)
            # with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
            #   pkl.dump(self.history, f)
            # self.test(sess, validate=True)
        sess.run(self._global_epoch.assign_add(1.))
	print('Average train per sec: ', avg_train_time/div_count)
	print('Average validation per sec: ', avg_val_time/div_count)
    except KeyboardInterrupt:
      try:
        raw_input('\nPress <Enter> to save or <Ctrl-C> to exit.')
      except:
        print('\r', end='')
        sys.exit(0)
    # saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
    #            latest_filename=self.name.lower(),
    #            global_step=self.global_epoch,
    #            write_meta_graph=False)
    # with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
    #   pkl.dump(self.history, f)
    # with open(os.path.join(self.save_dir, 'scores.txt'), 'a') as f:
    #   pass
    self.test(sess, validate=True)
    return


  def convert_bilou(self, indices):
    strings = map(lambda i: self._vocabs[3][i], indices)
    converted = []
    started_types = []
    # print(strings)
    for i, s in enumerate(strings):
      label_parts = s.split('/')
      curr_len = len(label_parts)
      combined_str = ''
      Itypes = []
      Btypes = []
      for idx, label in enumerate(label_parts):
        bilou = label[0]
        label_type = label[2:]
        props_str = ''
        if bilou == 'I':
          Itypes.append(label_type)
          props_str = ''
        elif bilou == 'O':
          curr_len = 0
          props_str = ''
        elif bilou == 'U':
          # need to check whether last one was ended
          props_str = '(' + label_type + ('*)' if idx == len(label_parts) - 1 else "")
        elif bilou == 'B':
          # need to check whether last one was ended
          props_str = '(' + label_type
          started_types.append(label_type)
          Btypes.append(label_type)
        elif bilou == 'L':
          props_str = ')'
          started_types.pop()
          curr_len -= 1
        combined_str += props_str
      while len(started_types) > curr_len:
        converted[-1] += ')'
        started_types.pop()
      while len(started_types) < len(Itypes) + len(Btypes):
        combined_str = '(' + Itypes[-1] + combined_str
        started_types.append(Itypes[-1])
        Itypes.pop()
      if not combined_str:
        combined_str = '*'
      elif combined_str[0] == "(" and combined_str[-1] != ")":
        combined_str += '*'
      elif combined_str[-1] == ")" and combined_str[0] != "(":
        combined_str = '*' + combined_str
      converted.append(combined_str)
    while len(started_types) > 0:
      converted[-1] += ')'
      started_types.pop()
    return converted

  def parens_check(self, srl_preds_str):
    for srl_preds in srl_preds_str:
      parens_count = 0
      for pred in srl_preds:
        for c in pred:
          if c == '(':
            parens_count += 1
          if c == ')':
            parens_count -= 1
            if parens_count < 0:
              return False
      if parens_count != 0:
        return False
    return True
    
  #=============================================================
  # TODO make this work if lines_per_buff isn't set to 0
  def test(self, sess, viterbi=False, validate=False):
    """"""
    
    if validate:
      filename = self.valid_file
      minibatches = self.valid_minibatches
      dataset = self._validset
      op = self.ops['test_op'][:15]
    else:
      filename = self.test_file
      minibatches = self.test_minibatches
      dataset = self._testset
      op = self.ops['test_op'][15:]
    
    all_predictions = [[]]
    all_sents = [[]]
    bkt_idx = 0
    total_time = 0.
    roots_lt_total = 0.
    roots_gt_total = 0.
    cycles_2_total = 0.
    cycles_n_total = 0.
    not_tree_total = 0.
    srl_correct_total = 0.
    srl_count_total = 0.
    forward_total_time = 0.
    non_tree_preds_total = []
    attention_weights = {}
    attn_correct_counts = {}
    pos_correct_total = 0.
    n_tokens = 0.
    for batch_num, (feed_dict, sents) in enumerate(minibatches()):
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      forward_start = time.time()
      probs, n_cycles, len_2_cycles, srl_probs, srl_preds, srl_logits, srl_correct, srl_count, srl_trigger, srl_trigger_targets, transition_params, attn_weights, attn_correct, pos_correct, pos_preds = sess.run(op, feed_dict=feed_dict)
      forward_total_time += time.time() - forward_start
      preds, parse_time, roots_lt, roots_gt, cycles_2, cycles_n, non_trees, non_tree_preds, n_tokens_batch = self.model.validate(mb_inputs, mb_targets, probs, n_cycles, len_2_cycles, srl_preds, srl_logits, srl_trigger, srl_trigger_targets, pos_preds, transition_params if viterbi else None)
      n_tokens += n_tokens_batch
      for k, v in attn_weights.iteritems():
        attention_weights["b%d:layer%d" % (batch_num, k)] = v
      for k, v in attn_correct.iteritems():
        if k not in attn_correct_counts:
          attn_correct_counts[k] = 0.
        attn_correct_counts[k] += v
      total_time += parse_time
      roots_lt_total += roots_lt
      roots_gt_total += roots_gt
      cycles_2_total += cycles_2
      cycles_n_total += cycles_n
      not_tree_total += non_trees
      srl_correct_total += srl_correct
      srl_count_total += srl_count
      pos_correct_total += pos_correct
      non_tree_preds_total.extend(non_tree_preds)
      all_predictions[-1].extend(preds)
      all_sents[-1].extend(sents)
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])

    correct = {'UAS': 0., 'LAS': 0., 'parse_eval': '', 'F1': 0.}
    srl_acc = 0.0
    if self.eval_parse:
      print("Total time in prob_argmax: %f" % total_time)
      print("Total time in forward: %f" % forward_total_time)
      print("Not tree: %d" % not_tree_total)
      print("Roots < 1: %d; Roots > 1: %d; 2-cycles: %d; n-cycles: %d" % (roots_lt_total, roots_gt_total, cycles_2_total, cycles_n_total))
      # ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
      # FORM: Word form or punctuation symbol.
      # LEMMA: Lemma or stem of word form.
      # UPOSTAG: Universal part-of-speech tag.
      # XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
      # FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
      # HEAD: Head of the current word, which is either a value of ID or zero (0).
      # DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
      # DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
      # MISC: Any other annotation.

      parse_gold_fname = self.gold_dev_parse_file if validate else self.gold_test_parse_file

      # write predicted parse
      parse_pred_fname = os.path.join(self.save_dir, "parse_preds.tsv")
      with open(parse_pred_fname, 'w') as f:
        for bkt_idx, idx in dataset._metabucket.data:
          data = dataset._metabucket[bkt_idx].data[idx]
          preds = all_predictions[bkt_idx][idx]
          words = all_sents[bkt_idx][idx]
          # sent[:, 6] = targets[tokens, 0] # 5 targets[0] = gold_tag
          # sent[:, 7] = parse_preds[tokens]  # 6 = pred parse head
          # sent[:, 8] = rel_preds[tokens]  # 7 = pred parse label
          # sent[:, 9] = targets[tokens, 1]  # 8 = gold parse head
          # sent[:, 10] = targets[tokens, 2]  # 9 = gold parse label
          sent_len = len(words)
          if self.eval_single_token_sents or sent_len > 1:
            for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
              head = pred[7] + 1
              tok_id = i + 1
              assert self.tags[datum[5]] == self.tags[pred[6]]
              tup = (
                tok_id,  # id
                word,  # form
                self.tags[pred[6]],  # gold tag
                # self.tags[pred[11]] if self.joint_pos_predicates or self.train_pos else self.tags[pred[4]], # pred tag or auto tag
                str(head if head != tok_id else 0),  # pred head
                self.rels[pred[8]] # pred label
              )
              f.write('%s\t%s\t_\t%s\t_\t_\t%s\t%s\n' % tup)
            f.write('\n')

      with open(os.devnull, 'w') as devnull:
        try:
          parse_eval = check_output(["perl", "bin/eval.pl", "-g", parse_gold_fname, "-s", parse_pred_fname], stderr=devnull)
          short_str = parse_eval.split('\n')[:3]
          print('\n'.join(short_str))
          print('\n')
          correct['parse_eval'] = parse_eval
          correct['LAS'] = short_str[0].split()[9]
          correct['UAS'] = short_str[1].split()[9]
        except CalledProcessError as e:
          print("Call to parse eval failed: %s" % e.output)

      if self.eval_by_domain:
        parse_gold_fname_path = '/'.join(parse_gold_fname.split('/')[:-1])
        parse_gold_fname_end = parse_gold_fname.split('/')[-1]
        for d in self._vocabs[5].keys():
          if d not in self._vocabs[5].SPECIAL_TOKENS:
            domain_gold_fname = os.path.join(parse_gold_fname_path, d + '_' + parse_gold_fname_end)
            domain_fname = os.path.join(self.save_dir, '%s_parse_preds.tsv' % d)
            with open(domain_fname, 'w') as f:
              for bkt_idx, idx in dataset._metabucket.data:
                data = dataset._metabucket[bkt_idx].data[idx]
                preds = all_predictions[bkt_idx][idx]
                words = all_sents[bkt_idx][idx]
                domain = '-'
                sent_len = len(words)
                if self.eval_single_token_sents or sent_len > 1:
                  for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
                    domain = self._vocabs[5][pred[5]]
                    head = pred[7] + 1
                    tok_id = i + 1
                    if domain == d:
                      tup = (
                        tok_id,  # id
                        word,  # form
                        self.tags[pred[6]],  # gold tag
                        # self.tags[pred[11]] if self.joint_pos_predicates or self.train_pos else self.tags[pred[4]], # pred tag or auto tag
                        str(head if head != tok_id else 0),  # pred head
                        self.rels[pred[8]]  # pred label
                      )
                      f.write('%s\t%s\t_\t%s\t_\t_\t%s\t%s\n' % tup)
                  if domain == d:
                    f.write('\n')
            with open(os.devnull, 'w') as devnull:
              try:
                parse_eval_d = check_output(["perl", "bin/eval.pl", "-g", domain_gold_fname, "-s", domain_fname],
                                          stderr=devnull)
                short_str_d = map(lambda s: "%s %s" % (d, s), parse_eval_d.split('\n')[:3])
                print('\n'.join(short_str_d))
                print('\n')
                # correct['parse_eval'] = parse_eval
                # correct['LAS'] = short_str[0].split()[9]
                # correct['UAS'] = short_str[1].split()[9]
              except CalledProcessError as e:
                print("Call to eval failed: %s" % e.output)

    if self.eval_srl:
      # load the real gold preds file
      srl_gold_fname = self.gold_dev_props_file if validate else self.gold_test_props_file

      # save SRL gold output for debugging purposes
      # srl_gold_write_fname = os.path.join(self.save_dir, 'srl_golds.tsv')
      # with open(srl_gold_write_fname, 'w') as f:
      #   for bkt_idx, idx in dataset._metabucket.data:
      #     # for each word, if trigger print word, otherwise -
      #     # then all the SRL labels
      #     data = dataset._metabucket[bkt_idx].data[idx]
      #     preds = all_predictions[bkt_idx][idx]
      #     words = all_sents[bkt_idx][idx]
      #     num_gold_srls = preds[0, 9]
      #     num_pred_srls = preds[0, 10]
      #     srl_preds = preds[:, 11 + num_pred_srls:11 + num_pred_srls + num_gold_srls]
      #     srl_preds_str = map(list, zip(*[self.convert_bilou(j) for j in np.transpose(srl_preds)]))
      #     # print(srl_preds_str)
      #     for i, (datum, word) in enumerate(zip(data, words)):
      #       pred = srl_preds_str[i] if srl_preds_str else []
      #       word_str = word if np.any(["(V*" in p for p in pred]) else '-'
      #       fields = (word_str,) + tuple(pred)
      #       owpl_str = '\t'.join(fields)
      #       f.write(owpl_str + "\n")
      #     f.write('\n')

      # save SRL gold output for debugging purposes
      srl_sanity_fname = os.path.join(self.save_dir, 'srl_sanity.tsv')
      with open(srl_sanity_fname, 'w') as f, open(filename, 'r') as orig_f:
        for bkt_idx, idx in dataset._metabucket.data:
          # for each word, if trigger print word, otherwise -
          # then all the SRL labels
          data = dataset._metabucket[bkt_idx].data[idx]
          preds = all_predictions[bkt_idx][idx]
          words = all_sents[bkt_idx][idx]
          num_gold_srls = preds[0, 12]
          num_pred_srls = preds[0, 13]
          srl_preds = preds[:, 14+num_pred_srls+num_gold_srls:]
          srl_golds = preds[:, 14+num_pred_srls:14+num_gold_srls+num_pred_srls]
          srl_preds_bio = map(lambda p: self._vocabs[3][p], srl_preds)
          srl_preds_str = map(list, zip(*[self.convert_bilou(j) for j in np.transpose(srl_preds)]))
          # todo if you want golds in here get it from the props file
          # srl_golds_str = map(list, zip(*[self.convert_bilou(j) for j in np.transpose(srl_golds)]))
          # print(srl_golds_str)
          # print(srl_preds_str)
          for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
            orig_line = orig_f.readline().strip()
            while not orig_line:
              orig_line = orig_f.readline().strip()
            orig_split_line = orig_line.split('\t')
            docid = orig_split_line[0]
            sentid = orig_split_line[1]
            domain = self._vocabs[5][pred[5]]
            orig_pred = srl_preds_str[i] if srl_preds_str else []
            # gold_pred = srl_golds_str[i] if srl_golds_str else []
            bio_pred = srl_preds_bio[i] if srl_preds_bio else []
            word_str = word
            tag0_str = self.tags[pred[6]] # gold tag
            tag1_str = self.tags[pred[8]] # auto tag
            tag2_str = self.tags[pred[11]] # predicted tag
            # gold_pred = word if np.any(["(V*" in p for p in gold_pred]) else '-'
            pred_pred = word if np.any(["(V*" in p for p in orig_pred]) else '-'
            # fields = (domain,) + (word_str,) + (tag0_str,) + (tag1_str,) + (tag2_str,) + (gold_pred,) + (pred_pred,) + tuple(bio_pred) + tuple(orig_pred)
            fields = (docid,) + (sentid,) + (word_str,) + (tag0_str,) + (tag1_str,) + (tag2_str,) + (pred_pred,) + tuple(bio_pred) + tuple(orig_pred)
            owpl_str = '\t'.join(fields)
            f.write(owpl_str + "\n")
          f.write('\n')

      # save SRL output
      srl_preds_fname = os.path.join(self.save_dir, 'srl_preds.tsv')
      with open(srl_preds_fname, 'w') as f:
        for bkt_idx, idx in dataset._metabucket.data:
          # for each word, if trigger print word, otherwise -
          # then all the SRL labels
          data = dataset._metabucket[bkt_idx].data[idx]
          preds = all_predictions[bkt_idx][idx]
          words = all_sents[bkt_idx][idx]
          num_gold_srls = preds[0, 12]
          num_pred_srls = preds[0, 13]
          srl_preds = preds[:, 14+num_gold_srls+num_pred_srls:]
          trigger_indices = preds[:, 14:14+num_pred_srls]
          srl_preds_str = map(list, zip(*[self.convert_bilou(j) for j in np.transpose(srl_preds)]))
          for i, (datum, word) in enumerate(zip(data, words)):
            pred = srl_preds_str[i] if srl_preds_str else []
            word_str = word if i in trigger_indices else '-'
            fields = (word_str,) + tuple(pred)
            owpl_str = '\t'.join(fields)
            f.write(owpl_str + "\n")
          if not self.parens_check(np.transpose(srl_preds_str)):
            print(np.transpose(srl_preds_str))
            print(map(lambda i: self._vocabs[3][i], np.transpose(srl_preds)))
          f.write('\n')

      srl_acc = (srl_correct_total / srl_count_total)*100.0

      with open(os.devnull, 'w') as devnull:
        try:
          srl_eval = check_output(["perl", "bin/srl-eval.pl", srl_gold_fname, srl_preds_fname], stderr=devnull)
          print(srl_eval)
          overall_f1 = float(srl_eval.split('\n')[6].split()[-1])
          correct['F1'] = overall_f1
        except CalledProcessError as e:
          print("Call to eval failed: %s" % e.output)

      if self.eval_by_domain:
        srl_gold_fname_path = '/'.join(srl_gold_fname.split('/')[:-1])
        srl_gold_fname_end = srl_gold_fname.split('/')[-1]
        for d in self._vocabs[5].keys():
          if d not in self._vocabs[5].SPECIAL_TOKENS:
            domain_gold_fname = os.path.join(srl_gold_fname_path, d + '_' + srl_gold_fname_end)
            domain_fname = os.path.join(self.save_dir, '%s_srl_preds.tsv' % d)
            with open(domain_fname, 'w') as f:
              for bkt_idx, idx in dataset._metabucket.data:
                # for each word, if trigger print word, otherwise -
                # then all the SRL labels
                data = dataset._metabucket[bkt_idx].data[idx]
                preds = all_predictions[bkt_idx][idx]
                words = all_sents[bkt_idx][idx]
                num_gold_srls = preds[0, 12]
                num_pred_srls = preds[0, 13]
                srl_preds = preds[:, 14 + num_gold_srls + num_pred_srls:]
                trigger_indices = preds[:, 14:14 + num_pred_srls]
                srl_preds_str = map(list, zip(*[self.convert_bilou(j) for j in np.transpose(srl_preds)]))
                domain = '-'
                for i, (datum, word, p) in enumerate(zip(data, words, preds)):
                  domain = self._vocabs[5][p[5]]
                  if domain == d:
                    pred = srl_preds_str[i] if srl_preds_str else []
                    word_str = word if i in trigger_indices else '-'
                    fields = (word_str,) + tuple(pred)
                    owpl_str = '\t'.join(fields)
                    f.write(owpl_str + "\n")
                if not self.parens_check(np.transpose(srl_preds_str)):
                  print(np.transpose(srl_preds_str))
                  print(map(lambda i: self._vocabs[3][i], np.transpose(srl_preds)))
                if domain == d:
                  f.write('\n')
            with open(os.devnull, 'w') as devnull:
              try:
                srl_eval_d = check_output(["perl", "bin/srl-eval.pl", domain_gold_fname, domain_fname], stderr=devnull)
                # print(srl_eval)
                str_d = srl_eval_d.split('\n')[6]
              except CalledProcessError as e:
                print("Call to eval failed: %s" % e.output)
                str_d = ""
            print("%sSRL %s:" % ("viterbi " if viterbi else "", d))
            print(str_d)

      # with open(os.path.join(self.save_dir, 'scores.txt'), 'a') as f:
      #   s, correct = self.model.evaluate(os.path.join(self.save_dir, os.path.basename(filename)), punct=self.model.PUNCT)
      #   f.write(s)

    if validate and self.multitask_layers != "":
      print("Attention UAS: ")
      multitask_uas_str = ''
      for k in sorted(attn_correct_counts):
        # todo w/ w/o mask punct
        attn_correct_counts[k] = attn_correct_counts[k] / n_tokens
        multitask_uas_str += '\t%s UAS: %.2f' % (k, attn_correct_counts[k] * 100)
      print(multitask_uas_str)

      if self.save_attn_weights:
        attention_weights = {str(k): v for k, v in attention_weights.iteritems()}
        np.savez(os.path.join(self.save_dir, 'attention_weights'), **attention_weights)

    pos_accuracy = (pos_correct_total/n_tokens)*100.0
    correct['POS'] = pos_accuracy
    # if validate:
    #   np.savez(os.path.join(self.save_dir, 'non_tree_preds.txt'), non_tree_preds_total)
    # print(non_tree_preds_total)
    # print(non_tree_preds_total, file=f)
    # las = np.mean(correct["LAS"]) * 100
    # uas = np.mean(correct["UAS"]) * 100
    print('UAS: %s    LAS: %s' % (correct["UAS"], correct["LAS"]))
    print('POS: %.2f' % pos_accuracy)
    print('SRL acc: %.2f' % (srl_acc))
    print('%sSRL F1: %s' % ("viterbi " if viterbi else "", correct["F1"]))
    return correct
  
  #=============================================================
  def savefigs(self, sess, optimizer=False):
    """"""
    
    import gc
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    matdir = os.path.join(self.save_dir, 'matrices')
    if not os.path.isdir(matdir):
      os.mkdir(matdir)
    for var in self.save_vars:
      if optimizer or ('Optimizer' not in var.name):
        print(var.name)
        mat = sess.run(var)
        if len(mat.shape) == 1:
          mat = mat[None,:]
        plt.figure()
        try:
          plt.pcolor(mat, cmap='RdBu')
          plt.gca().invert_yaxis()
          plt.colorbar()
          plt.clim(vmin=-1, vmax=1)
          plt.title(var.name)
          plt.savefig(os.path.join(matdir, var.name.replace('/', '-')))
        except ValueError:
          pass
        plt.close()
        del mat
        gc.collect()
    
  #=============================================================
  def _gen_ops(self):
    """"""
    
    # optims = {k: optimizers.RadamOptimizer(self._config, global_step=step) for k, step in self._global_steps}
    optimizer = optimizers.RadamOptimizer(self._config, global_step=self._global_step)
    train_output = self._model(self._trainset)

    # lrs = map(lambda o: o.learning_rate, optims)
    lr = optimizer.learning_rate

    train_op = optimizer.minimize(train_output['loss'])
    # train_ops = map(lambda k, o: o.minimize(train_output[k]), optims.iteritems())
    #
    # # ['pos_loss', 'trigger_loss', 'actual_parse_loss', 'srl_loss', 'multitask_loss_sum']
    # actual_train_ops = []
    # if self.train_pos:
    #   actual_train_ops.append(train_ops['pos_loss'])
    # if self.role_loss_penalty > 0:
    #   actual_train_ops.append(train_ops['srl_loss'])

    # These have to happen after optimizer.minimize is called
    valid_output = self._model(self._validset, moving_params=optimizer)
    test_output = self._model(self._testset, moving_params=optimizer)


    
    ops = {}
    # ops['train_op'] = actual_train_ops + [train_output['loss'],
    #                    train_output['n_correct'],
    #                    train_output['n_tokens']]
    # ops['train_op_svd'] = actual_train_ops + [train_output['loss'],
    #                        train_output['n_correct'],
    #                        train_output['n_tokens'],
    #                        train_output['roots_loss'],
    #                        train_output['2cycle_loss'],
    #                        train_output['svd_loss'],
    #                        train_output['log_loss'],
    #                        train_output['rel_loss']]
    # ops['train_op_srl'] = actual_train_ops + [train_output['loss'],
    #                        train_output['n_correct'],
    #                        train_output['n_tokens'],
    #                        train_output['roots_loss'],
    #                        train_output['2cycle_loss'],
    #                        train_output['svd_loss'],
    #                        train_output['log_loss'],
    #                        train_output['rel_loss'],
    #                        train_output['srl_loss'],
    #                        train_output['srl_correct'],
    #                        train_output['srl_count'],
    #                        train_output['trigger_loss'],
    #                        train_output['trigger_count'],
    #                        train_output['trigger_correct'],
    #                        train_output['pos_loss'],
    #                        train_output['pos_correct'],
    #                        train_output['multitask_losses']] + lrs
    ops['train_op'] = [train_op] + [train_output['loss'],
                       train_output['n_correct'],
                       train_output['n_tokens']]
    ops['train_op_svd'] = [train_op] + [train_output['loss'],
                           train_output['n_correct'],
                           train_output['n_tokens'],
                           train_output['roots_loss'],
                           train_output['2cycle_loss'],
                           train_output['svd_loss'],
                           train_output['log_loss'],
                           train_output['rel_loss']]
    ops['train_op_srl'] = [train_op] + [train_output['loss'],
                           train_output['n_correct'],
                           train_output['n_tokens'],
                           train_output['roots_loss'],
                           train_output['2cycle_loss'],
                           train_output['svd_loss'],
                           train_output['log_loss'],
                           train_output['rel_loss'],
                           train_output['srl_loss'],
                           train_output['srl_correct'],
                           train_output['srl_count'],
                           train_output['trigger_loss'],
                           train_output['trigger_count'],
                           train_output['trigger_correct'],
                           train_output['pos_loss'],
                           train_output['pos_correct'],
                           train_output['multitask_losses'],
                           lr]
    ops['valid_op'] = [valid_output['loss'],
                       valid_output['n_correct'],
                       valid_output['n_tokens'],
                       valid_output['predictions']]
    ops['test_op'] = [valid_output['probabilities'],
                      valid_output['n_cycles'],
                      valid_output['len_2_cycles'],
                      valid_output['srl_probs'],
                      valid_output['srl_preds'],
                      valid_output['srl_logits'],
                      valid_output['srl_correct'],
                      valid_output['srl_count'],
                      valid_output['srl_trigger'],
                      valid_output['srl_trigger_targets'],
                      valid_output['transition_params'],
                      valid_output['attn_weights'],
                      valid_output['attn_correct'],
                      valid_output['pos_correct'],
                      valid_output['pos_preds'],
                      test_output['probabilities'],
                      test_output['n_cycles'],
                      test_output['len_2_cycles'],
                      test_output['srl_probs'],
                      test_output['srl_preds'],
                      test_output['srl_logits'],
                      test_output['srl_correct'],
                      test_output['srl_count'],
                      test_output['srl_trigger'],
                      test_output['srl_trigger_targets'],
                      test_output['transition_params'],
                      test_output['attn_weights'],
                      test_output['attn_correct'],
                      test_output['pos_correct'],
                      test_output['pos_preds'],
                      ]
    # ops['optimizer'] = optimizer
    
    return ops
    
  #=============================================================
  # @property
  # def global_step(self):
  #   return self._global_step
  @property
  def global_epoch(self):
    return self._global_epoch
  @property
  def model(self):
    return self._model
  @property
  def words(self):
    return self._vocabs[0]
  @property
  def tags(self):
    return self._vocabs[1]
  @property
  def rels(self):
    return self._vocabs[2]
  @property
  def ops(self):
    return self._ops
  @property
  def save_vars(self):
    return self._save_vars
  
#***************************************************************
if __name__ == '__main__':
  """"""
  
  import argparse
  
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--test', action='store_true')
  argparser.add_argument('--load', action='store_true')
  argparser.add_argument('--model', default='Parser')
  argparser.add_argument('--matrix', action='store_true')
  argparser.add_argument('--profile', action='store_true')
  argparser.add_argument('--test_eval', action='store_true')
  
  args, extra_args = argparser.parse_known_args()
  cargs = {k: v for (k, v) in vars(Configurable.argparser.parse_args(extra_args)).iteritems() if v is not None}
  
  print('*** '+args.model+' ***')
  model = getattr(models, args.model)
  print(model)

  profile = args.profile
  
  # if 'save_dir' in cargs and os.path.isdir(cargs['save_dir']) and not (args.test or args.matrix or args.load):
  #   raw_input('Save directory already exists. Press <Enter> to overwrite or <Ctrl-C> to exit.')
  # if (args.test or args.load or args.matrix) and 'save_dir' in cargs:
  #   cargs['config_file'] = os.path.join(cargs['save_dir'], 'config.cfg')
  network = Network(model, **cargs)
  os.system('echo Model: %s > %s/MODEL' % (network.model.__class__.__name__, network.save_dir))

  # print variable names (but not the optimizer ones)
  print([v.name for v in network.save_vars if 'Optimizer' not in v.name and 'layer_norm' not in v.name])

  config_proto = tf.ConfigProto()
  config_proto.gpu_options.per_process_gpu_memory_fraction = network.per_process_gpu_memory_fraction

  # Create options to profile the time and memory information.
  if profile:
    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory()).order_by('micros').build()
  # Create a profiling context, set constructor argument `trace_steps`,
  # `dump_steps` to empty for explicit control.
  with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                        trace_steps=[],
                                        dump_steps=[]) if profile else dummy_context_mgr() as pctx:
    with tf.Session(config=config_proto) as sess:
      writer = tf.summary.FileWriter('./graphstb', sess.graph)
      sess.run(tf.global_variables_initializer())
      if not (args.test or args.matrix):
        if args.load:
          os.system('echo Training: > %s/HEAD' % network.save_dir)
          os.system('git rev-parse HEAD >> %s/HEAD' % network.save_dir)
          saver = tf.train.Saver(var_list=network.save_vars, save_relative_paths=True)
          saver.restore(sess, tf.train.latest_checkpoint(network.load_dir, latest_filename=network.name.lower()))
          if os.path.isfile(os.path.join(network.save_dir, 'history.pkl')):
            with open(os.path.join(network.save_dir, 'history.pkl')) as f:
              network.history = pkl.load(f)
        else:
          os.system('echo Loading: >> %s/HEAD' % network.load_dir)
          os.system('git rev-parse HEAD >> %s/HEAD' % network.save_dir)
        network.train(sess, profile)
      elif args.matrix:
        saver = tf.train.Saver(var_list=network.save_vars, save_relative_paths=True)
        saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
        # TODO make this save pcolor plots of all matrices to a directory in save_dir
        #with tf.variable_scope('RNN0/BiRNN_FW/LSTMCell/Linear', reuse=True):
        #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat0.pkl', 'w'))
        #with tf.variable_scope('RNN1/BiRNN_FW/LSTMCell/Linear', reuse=True):
        #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat1.pkl', 'w'))
        #with tf.variable_scope('RNN2/BiRNN_FW/LSTMCell/Linear', reuse=True):
        #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat2.pkl', 'w'))
        #with tf.variable_scope('MLP/Linear', reuse=True):
        #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat3.pkl', 'w'))
        network.savefigs(sess)
      else:
        os.system('echo Testing: >> %s/HEAD' % network.save_dir)
        os.system('git rev-parse HEAD >> %s/HEAD' % network.save_dir)
        saver = tf.train.Saver(var_list=network.save_vars, save_relative_paths=True)
        print("Loading model: ", network.load_dir)
        print(network.name.lower())
        saver.restore(sess, tf.train.latest_checkpoint(network.load_dir, latest_filename=network.name.lower()))

        # decode with & without viterbi
        network.test(sess, False, validate=True)
        network.test(sess, True, validate=True)

        # Actually evaluate on test data
        if args.test_eval:
          start_time = time.time()
          network.test(sess, network.viterbi_decode, validate=False)
          print('Parsing took %f seconds' % (time.time() - start_time))
    writer.close()

