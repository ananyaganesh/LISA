with open('saves/verbnet-predictions/vnroles.txt') as vnfile:
	count_dict = {}
	for line in vnfile:
		spline = line.split()
		count_dict[spline[0]] = int(spline[1])
	print(count_dict)
	
#with open('data/conll-2012-sdeps-filt-new/aligned-new.txt.bio') as conll_file:
#	o_counts = 0
#	for line in conll_file:
#		spline = line.split()
#		if len(spline) > 0:
#			ann = spline[1]
#			if bool(ann):
#				for string in spline:
#					if string == 'O':
#						o_counts += 1
#	count_dict['O'] = o_counts
#	total_sum = sum(x for x in count_dict.values())
#	print(total_sum)
#	print(float(count_dict['O'])/total_sum)
#	print(o_counts)

with open('saves/fixed-data/vnroles.txt') as old:
	oldsum = 0
	for line in old:
		oldsum += int(line.split()[1])
	print('Oldsum: ', oldsum)

with open('saves/verbnet-predictions/srls.txt') as new:
	newsum = 0
	for line in new:
		newsum += int(line.split()[1])
	print('Newsum: ', newsum)


