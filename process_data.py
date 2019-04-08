import sys
import numpy as np
import re

conll_fname = sys.argv[1]
semlink_fname = sys.argv[2]
out_fname = sys.argv[3]
try:
    docid_filt = sys.argv[4]
except IndexError:
    docid_filt = None

if docid_filt:
    with open(docid_filt, 'r') as f:
        docid_set = {line[:-1] for line in f}
else:
    docid_set = {}

print("using docids: ", docid_set)

arg_re = re.compile(r"(ARG[0-5A])-[A-Za-z]+")
remove_list = ['rel', 'LINK-SLC', 'LINK-PSV', 'LINK-PRO']
semlink_map = {}

with open(conll_fname) as conll_file:
    with open(out_fname, 'w') as out_file:
        buff = []
        oldkey = 'null'
        for conll_line in conll_file:
            split_conll_line = conll_line.split()

            # if line not blank, read all tokens in sentence
            if len(split_conll_line) > 0:

                if docid_set and split_conll_line[0] in docid_set:

                    # if new doc id, reset sentence num
                    if split_conll_line[0] != oldkey:
                        sentid = 0

                    # write sentence num into buffer for iding sentence later
                    split_conll_line.insert(2, str(sentid))
                    buff.append(split_conll_line)
                    oldkey = split_conll_line[0]

            # if end of sentence, process it
            elif buff:
                # identify sentence by doc id and sentence num
                conll_key = (buff[0][0], buff[0][2])

                # convert to np friendly format for convenience
                buff_array = np.array(buff, dtype=object)

                # all pred-arg SRL tags
                frames = buff_array[:, 15:-1]
                last_col = buff_array[:, -1]
                # column containing word
                words = buff_array[:, 4]
                # column containing root form of verb
                preds_col = buff_array[:, 10]

                num_preds = frames.shape[1]

                # collect non-empty predicates into array
                pred_array = np.empty((num_preds), dtype=object)
                for i in range(frames.shape[0]):
                    for j in range(frames.shape[1]):
                        if frames[i][j] == 'B-V':
                            pred_array[j] = preds_col[i]

                # mark if found in the semlink file
                annotated = False

                if conll_key[0].startswith('nw/wsj'):
                    semlink_buff = []
                    with open(semlink_fname) as semlink_file:
                        for semlink_line in semlink_file:
                            split_semlink_line = semlink_line.split()
                            semlink_id = split_semlink_line[0].split('.')
                            if semlink_id[1] == 'mrg':
                                continue

                            # identify each semlink entry by doc id and sentence num
                            semlink_key = (semlink_id[0], split_semlink_line[1])

                            # collect all matching entries for current sentence
                            if conll_key == semlink_key:
                                annotated = True
                                semlink_buff.append(split_semlink_line)

                    # print(conll_key, len(semlink_buff), lens)

                    # collect predicates and verbnet arguments from matched entries
                    sem_pred_list = []
                    sem_arg_list = []
                    for semlink_line in semlink_buff:
                        sem_pred = semlink_line[4].split('-')[0]
                        args = semlink_line[10:]
                        # take just the verbnet senses
                        stripped_args_vn = map(
                            lambda a: '-'.join(a.split('*')[-1].split('-')[1:]).split(';')[0].replace('-DSP', ''), args)

                        # want to replace all ARG[0-9A]-[az]+ with ARG[0-9A]
                        stripped_args = map(lambda a: arg_re.sub(r'\1', a), stripped_args_vn)

                        sem_args = [a for a in stripped_args if a not in remove_list]

                        sem_pred_list.append(sem_pred)
                        sem_arg_list.append(sem_args)

                    # keep track of how many preds have been aligned
                    counter = 0

                    # align each conll predicate-argument with semlink predicate-argument
                    for i in range(num_preds):
                        conll_pred = pred_array[i]
                        conll_args = frames[:, i]

                        # check all non-aligned semlink predicates for matches
                        for j in range(counter, len(sem_pred_list)):
                            sem_pred = sem_pred_list[j]
                            sem_args = sem_arg_list[j]
                            if conll_pred == sem_pred:
                                counter = j + 1

                                # if predicates match, start aligning arguments
                                for k, entry in enumerate(conll_args):

                                    # ignore verbs and outside tags
                                    if entry not in ['B-V', 'O']:

                                        # get ARG-x out of B-ARG-x, B-C-ARG-x, etc.
                                        if entry.startswith(('B-', 'I-')):
                                            if entry.startswith(('B-C-', 'I-C-', 'B-R-', 'I-R-')):
                                                conll_arg = entry[4:]
                                                # conll_arg = entry[:2] + entry[4:]
                                            else:
                                                conll_arg = entry[2:]
                                        else:
                                            conll_arg = entry

                                        #if '=' in conll_arg:
                                        #    print(num_preds, conll_key, conll_pred, i, j, k, conll_pred, conll_arg)

                                        # get propbank label and verbnet label from semlink arg
                                        for sem_arg in sem_args:
                                            if '=' in sem_arg:
                                                pb_part, vn_part = sem_arg.split('=')
                                            else:
                                                pb_part = sem_arg
                                                vn_part = None

                                                # non-annotated core args become 'Other'
                                                # if pb_part in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA']:
                                                #     vn_part = 'Other'

                                                # non-annotated non-core args fall back on propbank
                                                # else:
                                                # vn_part = pb_part

                                            # update the matching argument
                                            if conll_arg == pb_part:
                                                if vn_part:
                                                    frames[k, i] = vn_part + '=' + frames[k, i]
                                                break
                                # stop since a match has been found and aligned
                                break

                # write the sentence back with annotation and updated frames
                for i, token in enumerate(buff):
                    print(token[0] + '\t' + str(annotated) + '\t' + '\t'.join(token[1:15]) + '\t' + '\t'.join(frames[i]) + '\n')
                    out_file.write(token[0] + '\t' + str(annotated) + '\t' + '\t'.join(token[1:15]) + '\t' + '\t'.join(
                        frames[i]) + '\t' + last_col[i] + '\n')

                # reset buffer to get next sentence
                buff = []
                sentid += 1
                out_file.write('\n')
