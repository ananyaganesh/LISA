from collections import defaultdict
import numpy as np
with open('data/conll-2012-sdeps-filt-new/nw/nw-train-aligned.txt.bio','r') as conll_file:
	sent_counts = defaultdict(int)
	pred_counts = defaultdict(int)
	for line in conll_file:
		split_line = line.split()
		if len(split_line) > 0:
			key = (split_line[0], split_line[3])
			if split_line[1] == 'True':
				sent_counts[key] += 1
				if split_line[11] != '-':
					pred_counts[key] += 1
	tokens = np.array(sent_counts.values())
	preds = np.array(pred_counts.values())
	print('Annotated sents: ', len(sent_counts))
	print('Total tokens: ', np.sum(tokens))
	print('Max sent: ', np.max(tokens))
	print('Min sent: ', np.min(tokens), np.argmin(tokens))
	print('Num preds: ', np.sum(preds), len(pred_counts))
	for key,value in sent_counts.iteritems():
		if key not in pred_counts:
			print key

