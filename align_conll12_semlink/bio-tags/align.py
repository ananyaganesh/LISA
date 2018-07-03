from __future__ import print_function
import re
import sys

remove_list = ['rel', 'LINK-SLC', 'LINK-PSV', 'LINK-PRO']

arg_re = re.compile(r"(ARG[0-5A])-[A-Za-z]+")

semlink_fname = sys.argv[1]
conll_fname = sys.argv[2]
out_fname = sys.argv[3]
outfile = open(out_fname, 'w')

semlink_map = {}
arg_mapping_counts = {}
arg_mappings = {}
proposition_count = 0

with open(conll_fname, 'r') as conll_file:
    sent_buff = []
    old_key = ('null', 'null')
    for conll_line in conll_file:
        conll_split_line = conll_line.split()
        conll_key = (conll_split_line[0], conll_split_line[1])
        if conll_key == old_key:
            sent_buff.append(conll_split_line)
        else:
            if len(sent_buff) == 0:
                sent_buff = [conll_split_line]
                continue
            #print(conll_key, old_key)
            mat = []
            for sent_word in sent_buff:
                mat.append(sent_word[14:])
            tmat = list(zip(*mat))

            # find arguments of all predicates
            pred_arg = []
            for i, row in enumerate(tmat):
                arg_dict = {}
                #print(' ')
                for j, col in enumerate(row):
                    entry = tmat[i][j]
                    #print(entry, end=' ')
                    split_entry = entry.split('-')
                    if len(split_entry) == 1:
                        continue
                    if split_entry[1] == 'V':
                        arg_dict[tmat[i][j]] = sent_buff[j][10]
                    elif split_entry[1].startswith('ARG'):
                        if split_entry[1] in arg_dict:
							arg_dict[split_entry[1]].append((i,j))
                        else:
                            arg_dict[split_entry[1]] = [(i,j)]
                pred_arg.append(arg_dict)
                #print(pred_arg)
            
            #print(pred_arg)
            with open(semlink_fname, 'r') as semlink_file:
                for line in semlink_file:
                    line = line.strip()
                    if line:
                        proposition_count += 1
                        split_line = line.split()

                        # key is doc name without ending + sentence number
                        key = (split_line[0].split('.')[0], split_line[1])
                        if key == old_key:

							# value is predicate + args
							args = split_line[10:]
							# take just the verbnet senses
							stripped_args_vn = map(
								lambda a: '-'.join(a.split('*')[-1].split('-')[1:]).split(';')[0].replace('-DSP', ''),
								args)

							# verbnet and framenet senses
							stripped_args_fn = map(
								lambda a: '-'.join(a.split('*')[-1].split('-')[1:]).replace('-DSP', ''), args)

							# want to replace all ARG[0-9A]-[az]+ with ARG[0-9A]
							stripped_args = map(lambda a: arg_re.sub(r'\1', a), stripped_args_vn)

							stripped_removed_args = [a for a in stripped_args if a not in remove_list]

							value = (split_line[7].split('.')[0], ' '.join(stripped_removed_args))
							if key not in semlink_map:
								semlink_map[key] = []
							semlink_map[key].append(value)
                        
							sl_pred = split_line[4].split('-')[0]
							sl_args = stripped_removed_args

							#print(key, sl_pred, sl_args, pred_arg)

							for arg_dict in pred_arg:
								try:
									if arg_dict['B-V'] == sl_pred:
										#print(key, sl_pred, sl_args, arg_dict)
										for k, arg in enumerate(arg_dict.keys()):
											stripped_arg = arg.strip('(')
											for semlink_arg in sl_args:
												try:
													#print(semlink_arg, arg)
													if arg == semlink_arg.split('=')[0]:
														ind_list = arg_dict[arg]
														for ind in ind_list:
															#print(ind, mat[ind[1]][ind[0]])
															mat[ind[1]][ind[0]] = semlink_arg.split('=')[1] + '=' + mat[ind[1]][ind[0]]
														#print(key, sl_pred, semlink_arg, arg)
													else:
														pass
														#print(arg, semlink_arg.split('=')[0])
												except Exception as e:
													#print(e)
													continue
								except:
									continue

            for k, word_line in enumerate(sent_buff):
                outfile.write('\t'.join(word_line[:14]) + '\t' + '\t'.join(mat[k]) + '\n')
                #print('\t'.join(word_line[:14]) + '\t'.join(mat[k]))
            old_key = conll_key
            sent_buff = [conll_split_line]

# with open(semlink_fname, 'r') as semlink_file:
#     for line in semlink_file:
#         line = line.strip()
#         if line:
#             proposition_count += 1
#             split_line = line.split()
#
#             # key is doc name without ending + sentence number
#             key = (split_line[0].split('.')[0], split_line[1])
#
#             # value is predicate + args
#             args = split_line[10:]
#             # take just the verbnet senses
#             stripped_args_vn = map(
#                 lambda a: '-'.join(a.split('*')[-1].split('-')[1:]).split(';')[0].replace('-DSP', ''), args)
#
#             # verbnet and framenet senses
#             stripped_args_fn = map(lambda a: '-'.join(a.split('*')[-1].split('-')[1:]).replace('-DSP', ''), args)
#
#             # want to replace all ARG[0-9A]-[az]+ with ARG[0-9A]
#             stripped_args = map(lambda a: arg_re.sub(r'\1', a), stripped_args_vn)
#
#             stripped_removed_args = [a for a in stripped_args if a not in remove_list]
#
#             # update mapping counts
#             for arg in stripped_removed_args:
#                 if arg not in arg_mapping_counts:
#                     arg_mapping_counts[arg] = 0
#                 arg_mapping_counts[arg] += 1
#                 if '=' in arg:
#                     pb_arg, vn_arg = arg.split('=')
#                 else:
#                     pb_arg, vn_arg = arg, arg
#                 if pb_arg not in arg_mappings:
#                     arg_mappings[pb_arg] = {}
#                 if vn_arg not in arg_mappings[pb_arg]:
#                     arg_mappings[pb_arg][vn_arg] = 0
#                 arg_mappings[pb_arg][vn_arg] += 1
#
#             value = (split_line[7].split('.')[0], ' '.join(stripped_removed_args))
#             if key not in semlink_map:
#                 semlink_map[key] = []
#             semlink_map[key].append(value)

# print("Loaded %d semlink propositions" % proposition_count)
# # print(arg_mapping_counts)
# for arg in arg_mappings:
#     print("%s: %s" % (arg, arg_mappings[arg]))
