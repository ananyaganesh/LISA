''' Full suite of error analysis on CoNLL05 '''
from collections import Counter
import itertools
from itertools import izip
import subprocess
import sys
import re

from wsj_syntax_helper import extract_gold_syntax_spans
from full_analysis import read_file, read_conll_prediction, extract_spans,\
                          find, unlabeled_find

CORE_ROLES = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'AA']    
arg_dict = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
label_errors = 0
MAX_LEN = 200
CONLL05_GOLD_SYNTAX = 'conll12-syntax.txt'
CONLL05_GOLD_SRL = 'conll2012-dev-gold-props.txt'
#CONLL12_GOLD_SYNTAX = ''
#CONLL12_GOLD_SRL = 'conll2012-dev-gold-props.txt'


def fix_labels(pred_spans, gold_spans):
  ''' Change the label of an argument if its boundaries match the gold.
  '''
  ops = []
  new_spans = []
  for p in pred_spans:
    fixed = False
    for g in gold_spans:
      #print(p, g)
      if p[0] != g[0] and p[1] == g[1] and p[2] == g[2]:
        ops.append(("fix_label", p[0], g[0]))
        if g[0] in arg_dict:
          arg_dict[g[0]] += 1
        global label_errors
        label_errors += 1
        new_spans.append([g[0], p[1], p[2]])
        fixed = True
        break
    if not fixed:
      new_spans.append([p[0], p[1], p[2]])

  #print('Core arg errors: ', arg_dict)
  #print('Label_errors: ', label_errors)
  #core_errors = sum(arg_dict.values())
  #try:
    #print('Error proportion: ', float(core_errors)/label_errors)
  #except:
    #print('error')
  return new_spans, ops

def merge_two_spans(pred_spans, gold_spans, max_gap = 1):
  merged = [False] * len(pred_spans)
  ops = []
  new_spans = []
  for i, p1 in enumerate(pred_spans):
    for j, p2 in enumerate(pred_spans):
      if p1[2] < p2[1] and p1[2] + max_gap + 1 >= p2[1]:
        for g in gold_spans:
          if p1[1] == g[1] and p2[2] == g[2]:
            ops.append(("merge_two", p1, p2, g))
            new_spans.append([g[0], g[1], g[2]])
            merged[i] = True
            merged[j] = True
            break
      if merged[j]: continue
  #
  for i, p in enumerate(pred_spans):
    if not merged[i]:
      new_spans.append([p[0], p[1], p[2]])

  return new_spans, ops

def split_into_two_spans(pred_spans, gold_spans, max_gap = 1):
  ops = []
  new_spans = []
  for p in pred_spans:
    has_split = False
    for g1 in gold_spans:
      for g2 in gold_spans:
        if g1[2] < g2[1] and g1[2] + max_gap + 1 >= g2[1] and p[1] == g1[1] and g2[2] == p[2]:
          ops.append(("split_into_two", p, g1, g2))
          new_spans.append([g1[0], g1[1], g1[2]])
          new_spans.append([g2[0], g2[1], g2[2]])
          has_split = True
          break
      if has_split: break
    if not has_split:
      new_spans.append([p[0], p[1], p[2]])

  return new_spans, ops

def fix_left_boundary(pred_spans, gold_spans):
  ops = []
  new_spans = []
  for p in pred_spans:
    fixed = False
    for g in gold_spans:
      if p[0] == g[0] and p[1] != g[1] and p[2] == g[2]:
        ops.append(("fix_left_boundary", p, g))
        new_spans.append(g)
        fixed = True
        break
    if not fixed:
      new_spans.append(p)

  return new_spans, ops

def fix_right_boundary(pred_spans, gold_spans):
  ops = []
  new_spans = []
  for p in pred_spans:
    fixed = False
    for g in gold_spans:
      if p[0] == g[0] and p[1] == g[1] and p[2] != g[2]:
        ops.append(("fix_right_boundary", p, g))
        new_spans.append(g)
        fixed = True
        break
    if not fixed:
      new_spans.append(p)

  return new_spans, ops

def has_overlap(trg, spans, excl = []):
  for s in spans:
    if excl != [] and s[0] == excl[0] and s[1] == excl[1] and s[2] == excl[2]:
      continue
    if (s[1] <= trg[1] and trg[1] <= s[2]) or (trg[1] <= s[1] and s[1] <= trg[2]):
      return True
  return False 

def fix_both_boundaries(pred_spans, gold_spans):
  ops = []
  new_spans = []
  for p in pred_spans:
    fixed = False
    for g in gold_spans:
      if p[0] == g[0] and (p[1] != g[1] or p[2] != g[2]) \
          and ((p[1] <= g[1] and g[1] <= p[2]) or (g[1] <= p[1] and p[1] <= g[2])) \
          and not has_overlap(g, pred_spans, excl=p): 
        ops.append(("fix_right_boundary", p, g))
        new_spans.append(g)
        fixed = True
        break
    if not fixed:
      new_spans.append(p)

  return new_spans, ops

def move_core_arg(pred_spans, gold_spans):
  ops = []
  new_spans = []
  for p in pred_spans:
    moved = False
    if p[0] in CORE_ROLES:
      for g in gold_spans:
        if p[0] == g[0] and (p[2] < g[1] or p[1] > g[2]) \
          and len([g1 for g1 in gold_spans if g1[0] == g[0]]) == 1\
          and not has_overlap(g, pred_spans, excl=p): 
            ops.append(("move_core_arg", p, g))
            new_spans.append(g)
            moved = True
            break
    if not moved:
      new_spans.append(p)

  return new_spans, ops 

def drop_argument(pred_spans, gold_spans):
  ops = []
  new_spans = []
  for p in pred_spans:
    if not has_overlap(p, gold_spans):
      ops.append(("drop_arg", p))
    else:
      new_spans.append(p)
  return new_spans, ops


def add_argument(pred_spans, gold_spans):
  ops = []
  new_spans = []
  for g in gold_spans:
    if not has_overlap(g, pred_spans):
      ops.append(("add_arg", g))
      new_spans.append(g)

  for p in pred_spans:  
    new_spans.append(p)

  return new_spans, ops

def compute_pp_accuracy(pred_spans, gold_spans, syn_spans, words):
  num_pp_cases = 0
  num_correct_decisions = 0
  # 1. gold srl span contains a pp span
  for g in gold_spans:
    for s in syn_spans:
      #print g,s
      if "PP" in s[0].split('-') and g[1] < s[1] and s[2] == g[2] \
          and words[s[1]] != 'of':
        #print g, s, ' '.join(words[g[1]: g[2] + 1])
        num_pp_cases += 1
        for p in pred_spans:
          if p[1] == g[1] and p[2] == g[2]:
            num_correct_decisions += 1
            break
        #
    #
    #for s1 in syn_spans:
    #  if "PP" in s1[0].split('-') and g[2] < s[1]: 
    #    for s2 in syn_spans:
  return num_correct_decisions, num_pp_cases

def compute_f1(num_matched, num_predicted, num_gold):
  precision = 100.0 * num_matched / num_predicted
  recall = 100.0 * num_matched / num_gold
  f1 = 2.0 * precision * recall / (precision + recall)
  return precision, recall, f1

def update_confusion_matrix(cmat, ops):
  for op in ops:
    p = op[1]
    g = op[2]
    if not p in cmat:
      cmat[p] = {}
    if not g in cmat[p]:
      cmat[p][g] = 1
    else:
      cmat[p][g] += 1

def get_syn_span(span, syn_spans):
  for s in syn_spans:
    if span[1] == s[1] and span[2] == s[2]:
      return s
  return []


if __name__ == '__main__':
  #sentences, gold, predicted = read_file(sys.argv[1])
  sentences, predicates, gold = read_conll_prediction(CONLL05_GOLD_SRL)
  #print(sentences[0], predicates[0], gold[0])
  _, pred_predicates, predicted = read_conll_prediction(sys.argv[1])
  words, postags, syn_spans = extract_gold_syntax_spans(CONLL05_GOLD_SYNTAX)

  #print len(sentences), len(predicates), len(syn_spans), len(gold), len(predicted)
  #print(len(words))
  assert len(gold) == len(predicted)

  num_matched = 0
  num_gold = 0
  num_predicted = 0

  num_new_predicted = [0] * 10
  num_new_matched = [0] * 10

  num_pp_cases = 0
  num_correct_pps = 0

  confusion_matrix = {} # Pred->Gold
  arg_drop_by_label = Counter()
  arg_drop_by_dist = Counter()
  
  fun_ops = [fix_labels, move_core_arg, merge_two_spans, split_into_two_spans, fix_both_boundaries,\
              drop_argument, add_argument]
  #fun_ops = [fix_labels, add_argument]

  # Deep copy. 
  new_pred = []
  for s0, w0, props, p_props, p0 in izip(sentences, words, predicates, pred_predicates, predicted):
    if p_props == props:
      new_pred.append([])
      for prop_id, pred_args in izip(props, p0):
        #print prop_id, gold_args, pred_args
        new_pred[-1].append([s for s in pred_args if s[0] != 'V' and not 'C-V' in s[0]])
  
  num_pp_involved = 0
  num_merge_ops = 0
  num_split_ops = 0
  attachments = Counter()
  sem_attachments = Counter()
 
  for i, fun_op in enumerate(fun_ops):
    sid = 0
    f = open('temp.txt', 'w')
    for s0, w0, props, p_props, syn, g0, p0 in izip(sentences, words, predicates, pred_predicates, syn_spans, gold, predicted):
      if p_props == props:

        pid = 0
        for prop_id, gold_args, pred_args in izip(props, g0, p0):

          #print prop_id, gold_args, pred_args
          gold_spans = [s for s in gold_args if s[0] != 'V' and not 'C-V' in s[0]]
          pred_spans = [s for s in pred_args if s[0] != 'V' and not 'C-V' in s[0]]

          # Compute F1
          if i == 0:
            gold_matched = [find(g, pred_spans) for g in gold_spans]
            pred_matched = [find(p, gold_spans) for p in pred_spans]
            num_gold += len(gold_spans)
            num_predicted += len(pred_spans)
            num_matched += sum(pred_matched)
            matched_pairs = [["", p[0], p[0]] for p in pred_spans if find(p, gold_spans)]
            update_confusion_matrix(confusion_matrix, matched_pairs)

          #print len(new_pred), len(new_pred[sid]), sid, pid
          while (True):
            new_spans, ops = fun_op(new_pred[sid][pid], gold_spans)
            new_pred[sid][pid] = [p for p in new_spans]
            #print new_spans, '\n', gold_spans, '\n', ops
            if ops == []: break
            if ops[0][0] == "fix_label":
              update_confusion_matrix(confusion_matrix, ops)
            if ops[0][0] == "merge_two":
              num_merge_ops += len(ops)
              for op in ops:
                sem_attachments[op[2][0]] += 1
                ss = get_syn_span(op[2], syn)
                if ss != []:
                  attachments[ss[0].split('-')[0]] += 1
                if ss != [] and 'PP' in ss[0]:
                  num_pp_involved += 1
                  '''print ' '.join(w0)
                  print s0[prop_id]
                  print "Gold: ", gold_spans
                  print "Pred: ", pred_spans, '\n'''

            if ops[0][0] == "split_into_two":
              num_split_ops += len(ops)
              for op in ops:
                sem_attachments[op[3][0]] += 1
                ss = get_syn_span(op[3], syn)
                if ss != []:
                  attachments[ss[0].split('-')[0]] += 1
                if ss != [] and 'PP' in ss[0]:
                  num_pp_involved += 1

          #new_gold_matched = [find(g, new_spans) for g in gold_spans]
          new_pred_matched = [find(p, gold_spans) for p in new_spans]
          num_new_predicted[i] += len(new_spans)
          num_new_matched[i] += sum(new_pred_matched)
          pid += 1

        #
        # Write to file
        for t in range(len(s0)):
          if t in props:
            f.write(s0[t])
          else:
            f.write('-')
          for p, prop_id in enumerate(props):
            f.write('\t')
            if t == prop_id:
              f.write('B-V')
              continue
            in_span = False
            for s in new_pred[sid][p]:
              if s[1] == t:
                f.write('B-' + s[0])
                in_span = True
                break
              if s[1] <= t and t <= s[2]:
                f.write('I-' + s[0])
                in_span = True
                break
            if not in_span:
              f.write('O')
          f.write('\n')
        f.write('\n')

        sid += 1

    f.close()
    # Run eval script.
    '''child = subprocess.Popen('sh {} {} {}'.format('/home/luheng/Workspace/neural_srl/data/srl/conll05-eval.sh',\
                                                   '/home/luheng/Workspace/neural_srl/data/srl/conll05.devel.props.gold.txt',\
                                                    'temp.txt'),\
                              shell = True, stdout=subprocess.PIPE)
    eval_info = child.communicate()[0]
    try:
      Fscore = eval_info.strip().split("\n")[6]
      Fscore = Fscore.strip().split()[6]
      accuracy = float(Fscore)
      #print(eval_info)
      print("Fscore={}".format(accuracy))
      official_f1s.append(accuracy)
    except IndexError:
      print("Unable to get FScore. Skipping.")'''

  print "Original:"
  p,r,f1 = compute_f1(num_matched, num_predicted, num_gold)
  print "Precision: {}, Recall: {}, F1: {}".format(p, r, f1)
  prev_f1 = f1

  for i, fun_op in enumerate(fun_ops):
    print str(fun_op).split()[1]
    p, r, f1 = compute_f1(num_new_matched[i], num_new_predicted[i], num_gold)
    print "Precision: {}, Recall: {}, F1: {}, delta={}".format(p, r, f1, f1 - prev_f1)
    prev_f1 = f1

  print '\n'

  # Print confusion matrix
  row_keys = sorted(confusion_matrix.keys())
  col_keys = set([])
  freq = {}
  for p in row_keys:
    for g in confusion_matrix[p].keys():
      col_keys.add(g)
      if not p in freq:
        freq[p] = 0
      if not g in freq:
        freq[g] = 0
      freq[p] += 1
      freq[g] += 1

  row_keys = sorted([r for r in row_keys if freq[r] > 10])
  col_keys = sorted([c for c in col_keys if freq[c] > 10])
  if 'AM-EXT' in row_keys:
    row_keys.remove('AM-EXT')
  if 'AM-EXT' in col_keys:
    col_keys.remove('AM-EXT')

  col_normalizer = {}
  for g in col_keys:
    col_normalizer[g] = 0
    for p in row_keys:
      if g in confusion_matrix[p] and p != g:
        col_normalizer[g] += confusion_matrix[p][g]

  print '     \t&' + '\t&'.join([c.split('-')[-1] for c in col_keys]) #+ '\\\\'
  for p in row_keys:
    print p.split('-')[-1],
    for g in col_keys:
      if g in confusion_matrix[p] and p != g:
        print '\t& {:.0f}'.format(100.0 * confusion_matrix[p][g] / col_normalizer[g]),
      elif p == g:
        print '\t& -',
      else:
        print '\t& 0',
    print '\n',

  # Recall loss analysis
  print '\n'.join([str(a) for a in arg_drop_by_label.most_common(10)])
  print '\n'.join([str(a) for a in arg_drop_by_dist.most_common(10)])

  total = num_split_ops + num_merge_ops
  print "Num. split-merge ops: {}. Num. PPs involved: {}".format(num_split_ops + num_merge_ops, num_pp_involved)
  for label, freq in attachments.most_common(10):
    print "{}\t{}\t{}".format(label, freq, 100.0 * freq / total)

  print "Types of semantic arguments"
  for label, freq in sem_attachments.most_common(10):
    print "{}\t{}\t{}".format(label, freq, 100.0 * freq / total)
 
  print('Core arg arrors: ', arg_dict)
  print('Total core arg errors: ', sum(arg_dict.values()))
  print('Total label errors: ', label_errors)

