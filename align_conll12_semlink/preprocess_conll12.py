conll_fname = "conll2012-wsj-train.txt"
conll_outname = "conll2012-wsj-processed.txt"
with open(conll_fname, 'r') as conll_file:
	with open(conll_outname, 'w') as conll_out:
		docid = 'null'
		for line in conll_file:
			split_line = line.split()
			if split_line[0] != docid:
				sent_id = -1
				docid = split_line[0]
			if split_line[2] == '0':
				sent_id += 1
			conll_out.write('\t'.join(split_line[:1]) + '\t' + str(sent_id) + '\t' + '\t'.join(split_line[1:]) + '\n')

