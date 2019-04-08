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
from collections import Counter

from lib.etc.k_means import KMeans
from configurable import Configurable

from vocab import Vocab
from metabucket import Metabucket

#***************************************************************



class Dataset(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, filename, vocabs, mappings, builder, *args, **kwargs):
    """"""
    
    super(Dataset, self).__init__(*args, **kwargs)
    self.vocabs = vocabs

    self.train_domains_set = set(self.train_domains.split(',')) if self.train_domains != '-' and self.name == "Trainset" else set()
    print("Loading training data from domains:", self.train_domains_set if self.train_domains_set else "all")

    self._file_iterator = self.file_iterator(filename)
    self._train = (filename == self.train_file)
    self._metabucket = Metabucket(self._config, n_bkts=self.n_bkts)
    self._data = None
    self.mappings = mappings
    self.rebucket()

    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    self.targets = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='targets')
    self.srl_targets_pb = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='srl_targets_pb')
    self.srl_targets_vn = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='srl_targets_vn')
    self.annotated = tf.placeholder(dtype=tf.int32, shape=(None, None), name='annotated')
    self.step = tf.placeholder_with_default(0., shape=None, name='step')
    self.builder = builder()
  
  #=============================================================
  def file_iterator(self, filename):
    """"""
    
    with open(filename) as f:
      if self.lines_per_buffer > 0:
        buff = [[]]
        while True:
          line = f.readline()
          while line:
            line = line.strip().split()
            if line and (not self.train_domains_set or line[0].split('/')[0] in self.train_domains):
              buff[-1].append(line)
            else:
              if len(buff) < self.lines_per_buffer:
                if len(buff[-1]) > 0:
                  buff.append([])
                else:
                  buff[-1] = []
              else:
                break
            line = f.readline()
          if not line:
            f.seek(0)
          else:
            buff = self._process_buff(buff)
            yield buff
            line = line.strip().split()
            if line:
              buff = [[line]]
            else:
              buff = [[]]
      else:
        buff = [[]]
        for line in f:
          line = line.strip().split()
          if line and (not self.train_domains_set or line[0].split('/')[0] in self.train_domains):
            buff[-1].append(line)
          else:
            if len(buff[-1]) > 0:
              buff.append([])
            else:
              buff[-1] = []
        if buff[-1] == []:
          buff.pop()
        buff = self._process_buff(buff)
        while True:
          yield buff
  
  #=============================================================
  def _process_buff(self, buff):
    """"""

    # tmp_f = open("debug_data_%s" % self.name, 'w')
    
    words, tags, rels, srls, predicates, domains, vnroles = self.vocabs
    #print('SRL in vocabs: ', srls[0])
    srl_start_field = srls.conll_idx[0]
    sents = 0
    toks = 0
    examples = 0
    total_predicates = 0
    buff2 = []
    annotation = {'True': 1, 'False': 0}
    # process tokens in each sent, store in buff[i][j]
    for i, sent in enumerate(buff):
      # if not self.conll2012 or (self.conll2012 and len(list(sent)) > 1):
      #print(sent, len(sent))
      sents += 1
      sent_len = len(sent)
      num_fields = len(sent[0])

      # indices of SRLs from 14 to end of sentence
      #print('SRL start field: ', srl_start_field)
      ##
      srl_take_indices = [idx for idx in range(srl_start_field, srl_start_field + sent_len) if idx < num_fields - 1 and (self.train_on_nested or np.all(['/' not in sent[j][idx] for j in range(sent_len)]))]
      predicate_indices = []
      has_vn_anno = [False for _ in range(len(srl_take_indices))]
      for j, token in enumerate(sent):
        toks += 1
        if self.conll:
          word, tag1, tag2, head, rel = token[words.conll_idx], token[tags.conll_idx[0]], token[tags.conll_idx[1]], token[8], token[rels.conll_idx]
          if rel == 'root':
            head = j
          else:
            head = int(head) - 1
          buff[i][j] = (word,) + words[word] + tags[tag1] + tags[tag2] + (head,) + rels[rel]
        elif self.conll2012:
          word, auto_tag, gold_tag, head, rel, annotated = token[words.conll_idx], token[tags.conll_idx[0]], token[tags.conll_idx[1]], token[8], token[rels.conll_idx], token[1]
          domain = token[0].split('/')[0]
          #print(word, auto_tag, gold_tag, head, rel, annotated)
          if rel == 'root':
            head = j
          else:
            head = int(head) - 1

          # srl_fields = [token[idx] if idx < len(token)-1 else 'O' for idx in range(srl_start_field, srl_start_field + sent_len)]
          srl_fields_full = [token[idx] for idx in srl_take_indices] # todo can we use fancy indexing here?

          #print('SRL fields full: ', srl_fields_full)

          # split propbank from verbnet labels e.g. Agent=ARG0
          srl_vn_labels = [tuple(srl_str.split('=')) for srl_str in srl_fields_full]
          srl_fields = [srl_str[1] if len(srl_str) > 1 else srl_str[0] for srl_str in srl_vn_labels]

          #print('SRL fields: ', srl_fields, len(srl_fields))
          vn_fields = []
          for idx, srl_str in enumerate(srl_vn_labels):
            if len(srl_str) > 1:
              vn_fields.append(srl_str[0])
              has_vn_anno[idx] = True
            elif srl_str[0] == 'O':
              vn_fields.append('O')
            elif srl_str[0].split('-')[1] == 'V':
              vn_fields.append('V')
            # elif srl_str[0] == 'B-ARGA' or srl_str[0] == 'I-ARGA':
            #   vn_fields.append('ARGA')
            # elif srl_str[0].startswith(('B-ARGM', 'I-ARGM', 'B-R-ARGM', 'B-C-ARGM', 'I-R-ARGM', 'I-C-ARGM')):
            #   vn_fields.append('-'.join(srl_str[0].split('-')[-2:]))
            else:
              vn_fields.append('NoLabel')

          #vn_fields = [srl_str[0] if len(srl_str) > 1 else 'NoLabel' if srl_str[0] is not 'O' else 'O' for srl_str in srl_vn_labels]
          #print('VN fields: ', vn_fields, len(vn_fields))

          # srl_fields += ['O'] * (sent_len - len(srl_take_indices))
          srl_tags = [srls[s][0] for s in srl_fields]

          # vn_fields += ['O'] * (sent_len - len(srl_take_indices))
          vn_tags = [vnroles[s][0] for s in vn_fields]

          #print(word, vn_fields, srl_fields)

          if self.joint_pos_predicates:
            is_predicate = token[predicates.conll_idx[0]] != '-' and (self.train_on_nested or self.predicate_str in srl_fields)
            tok_predicate_str = str(is_predicate) + '/' + gold_tag
          else:
            is_predicate = token[predicates.conll_idx] != '-' and (self.train_on_nested or self.predicate_str in srl_fields)
            tok_predicate_str = str(is_predicate)

          if is_predicate:
            predicate_indices.append(j)

          buff[i][j] = (word,) + words[word] + tags[auto_tag] + predicates[tok_predicate_str] + domains[domain] + (sents,) + (annotation[annotated],) + tags[gold_tag] + (head,) + rels[rel] + tuple(srl_tags) + tuple(vn_tags)

      # go back and mark predicates that don't have a verbnet annotation with NoLabel
      for srl_idx, has_vn in enumerate(has_vn_anno):
        if not has_vn:
          buff[i][-1] = buff[i][j][:11+len(has_vn_anno)+srl_idx] + (vnroles['NoLabel'][0],) + buff[i][j][11+len(has_vn_anno)+srl_idx+1:]

      # Expand sentences into one example per predicate
      if self.one_example_per_predicate:
        # grab the sent
        # should be sent_len x sent_elements
        sent = np.array(buff[i])
        # print(sent)
        is_predicate_idx = 4
        srl_start_idx = 11
        #print(len(sent), len(sent)-srl_start_idx)
        word_part = sent[:, 0].astype('O')
        srl_part = sent[:, srl_start_idx:].astype(np.int32)
        rest_part = sent[:, 1:srl_start_idx].astype(np.int32)
        # print("orig sent (%d):" % len(predicate_indices), sent[:, :8+len(predicate_indices)])
        # print("orig preds:", [map(lambda x: srls[int(x)], t) for t in sent[:, srl_start_idx:srl_start_idx+len(predicate_indices)]])
        if predicate_indices:
          for k, p_idx in enumerate(predicate_indices):
            # should be sent_len x sent_elements
            rest_part[:, is_predicate_idx-1] = predicates["False"][0]
            rest_part[p_idx, is_predicate_idx-1] = predicates["True"][0]
            correct_srls = srl_part[:, k]
            new_sent = np.concatenate([np.expand_dims(word_part, -1), rest_part, np.expand_dims(correct_srls, -1)], axis=1)
            buff2.append(new_sent)
            # print("new sent:", new_sent)
            # print("new preds:", map(lambda x: srls[int(x)], new_sent[:, -1]))
            # tokens_str = ' '.join(word_part)
            # labels_str = ' '.join(map(lambda x: srls[x], correct_srls))
            ## idx, tokens, labels
            # print("%d %s ||| %s" % (p_idx, tokens_str, labels_str), file=tmp_f)
            total_predicates += 1
            examples += 1
        else:
           new_sent = np.concatenate([np.expand_dims(word_part, -1), rest_part], axis=1)
           buff2.append(new_sent)
           examples += 1
      # else:
      #   buff2.append(np.concatenate[np.expand_dims(word_part, -1), rest_part, srl_part], axis=1) #(sent[0],) + map(int, sent[1:]))
      #   examples += 1
    # tmp_f.close()
    #self.define_mappings()
    if self.one_example_per_predicate:
      print("Loaded %d sentences with %d tokens, %d examples (%d predicates) (%s)" % (sents, toks, examples, total_predicates, self.name))
      return buff2
    else:
      print("Loaded %d sentences with %d tokens (%s)" % (sents, toks, self.name))
      return buff
  
  #=============================================================
  def reset(self, sizes):
    """"""
    
    self._data = []
    self._targets = []
    print('Sizes of splits: ', sizes)
    self._metabucket.reset(sizes)
    return
  
  #=============================================================
  def rebucket(self):
    """"""

    buff = self._file_iterator.next()
    len_cntr = Counter()
    
    for sent in buff:
      len_cntr[len(sent)] += 1
    self.reset(KMeans(self.n_bkts, len_cntr).splits)
    
    for sent in buff:
      self._metabucket.add(sent)

    self._finalize()
    return
  
  #=============================================================
  def _finalize(self):
    """"""
    
    self._metabucket._finalize()
    return
  
  #=============================================================
  def get_minibatches(self, batch_size, input_idxs, target_idxs, shuffle=True):
    """"""
    minibatches = []
    for bkt_idx, bucket in enumerate(self._metabucket):
      #print('Index, bucket, length: ', bkt_idx, len(bucket), batch_size, bucket.size)
      if batch_size == 0:
        n_splits = 1
      else:
        n_tokens = len(bucket) * bucket.size
        n_splits = max(n_tokens // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      arr_sp = np.array_split(range_func(len(bucket)), n_splits)
      for bkt_mb in arr_sp:
        minibatches.append( (bkt_idx, bkt_mb) )
    if shuffle:
      np.random.shuffle(minibatches)

    for bkt_idx, bkt_mb in minibatches:
      #print(bkt_idx, bkt_mb)
      feed_dict = {}
      data = self[bkt_idx].data[bkt_mb]
      sents = self[bkt_idx].sents[bkt_mb]
      # length of maximum
      maxlen = np.max(np.sum(np.greater(data[:,:,0], 0), axis=1))

      #print('Data shape: ', data.shape)
      #print('Data[1].shape: ', data[0].shape)
      #print('Data[0][1] shape: ', data[0][0].shape)

      # np.set_printoptions(threshold=np.nan)
      # print("maxlen", maxlen)
      # print("maxlen+max(target_idxs)", maxlen+max(target_idxs))
      # print("data.shape[2]", data.shape[2])
      #print('inputs shape: ', data[:,:maxlen,input_idxs].shape)
      targets = data[:,:maxlen,min(target_idxs):maxlen+max(target_idxs)+1]
      #print("data shape", targets.shape)
      #print("data[:,:,3:] shape, maxlen: ", targets[:,:,3:].shape, maxlen+max(target_idxs)+1)

      #print('Data: ', data[0][0], len(data[0][0]))
      #print('Target: ', targets[0][0], len(targets[0][0]))
      #for item in data:
      #  print('Item size: ', item.shape)

      srl_total = data.shape[2] - 10
      srl_vn_start = 10 + srl_total // 2
      srl_vn = data[:, :maxlen, srl_vn_start:]
      srl_pb = targets[:,:,3:]
      # print('Sents len: ', len(sents))
      # for i,sentence in enumerate(data):
      #   print('Curr sent len: ', len(sents[i]))
      #   for j,token in enumerate(sentence):
      #     print('Annotation: ', token[6])
      #     try:
      #       print('Word: ', sents[i][j])
      #     except IndexError:
      #       print('No word')
      #     #print('SRL part: ', token[10: srl_vn_start], len(token[10: srl_vn_start]))
      #     print('VN part: ', token[srl_vn_start:], len(token[srl_vn_start:]))
      #print('SRL total: ', srl_total, 'SRL VN start: ', srl_vn_start)
      #print(targets[:,:,3:])
      #print(min(target_idxs), maxlen+max(target_idxs)+1)

      feed_dict.update({
        self.inputs: data[:, :maxlen, input_idxs],
        self.targets: data[:, :maxlen, min(target_idxs):],
        # self.targets: data[:,:maxlen,min(target_idxs):maxlen+max(target_idxs)+1],
        # self.srl_targets_pb: targets[:,:,3:],
        # self.srl_targets_vn: data[:, :maxlen, srl_vn_start:],
        self.annotated: data[:,:maxlen, 6]
      })

      annot = feed_dict[self.annotated]
      inp = feed_dict[self.inputs]
      # print('Annotation shape: ', annot.shape, 'Input shape: ', inp.shape)
      # print('Annotation: ', annot[0], 'Input: ', inp[0])
      yield feed_dict, sents
  
  #=============================================================
  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts

  
  #=============================================================
  def __getitem__(self, key):
    return self._metabucket[key]
  def __len__(self):
    return len(self._metabucket)
