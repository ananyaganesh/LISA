import sys
import numpy as np
from itertools import izip
from collections import defaultdict

preds_fname = sys.argv[1]
gold_fname = sys.argv[2]

with open(preds_fname) as preds_file, open(gold_fname) as gold_file:
    count = 0
    missed_preds = []
    extra_preds = []
    wrong_preds = []
    confusion_matrix = {}
    wlc = 0
    for pred_line, gold_line in izip(preds_file, gold_file):
        split_pred_line = pred_line.split()
        split_gold_line = gold_line.split()

        if len(split_gold_line) != len(split_pred_line):
            wlc += 1

        if len(split_pred_line) > 0 and len(split_gold_line) > 0:
            predpred = split_pred_line[0]
            goldpred = split_gold_line[0]
            if predpred == goldpred:
                print('Predicted: ', predpred, 'Gold: ', goldpred)
            if split_pred_line[0] == '-' and split_gold_line[0] != '-':
                missed_preds.append((count, split_gold_line[0]))
            elif split_pred_line[0] != '-' and split_gold_line[0] == '-':
                extra_preds.append((count, split_pred_line[0]))
            elif split_pred_line[0] != split_gold_line[0]:
                wrong_preds.append((count, split_pred_line[0], split_gold_line[0]))

            if len(split_gold_line) == len(split_pred_line):
                #print(split_gold_line[1:], split_pred_line[1:])
                for i in range(1, len(split_gold_line)):
                    goldarg = split_gold_line[i]
                    predarg = split_pred_line[i]
                    if goldarg not in confusion_matrix:
                        confusion_matrix[goldarg] = defaultdict(int)
                    else:
                        confusion_matrix[goldarg][predarg] += 1

        count += 1

    print('Missed: ', len(missed_preds))
    print('Extra: ', len(extra_preds))
    print('Wrong: ', len(wrong_preds))
    print('Wrong lengths: ', wlc)

    for key, value in confusion_matrix.iteritems():
        print(key, dict(confusion_matrix[key]))