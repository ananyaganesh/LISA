conll12_fname = "conll2012-wsj-processed.txt"

for count in range(2, 22):
	with open(conll12_fname, 'r') as conll12_file:
		if count < 10:
			ind = str(0) + str(count)
		else:
			ind = str(count)
		
		out_fname = "conll2012-wsj" + ind + ".txt"

		with open(out_fname, 'w') as out_file:
			for line in conll12_file:
				if line.startswith("nw/wsj/" + ind):
					out_file.write(line)

	

