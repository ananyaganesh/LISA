''' Extract gold syntactic constituents
'''

import sys, re

def extract_gold_syntax_spans(filename):
  fin = open(filename, 'r')

  sentences = []
  postags = []
  spans = []

  s0 = []
  p0 = []
  c0 = []
  stack = []

  for line in fin:
    line = line.strip()
    if line == '' and len(s0) > 0:
      #if len(sentences) < 5:
      #  print s0
      #  print c0

      sentences.append(s0)
      postags.append(p0)
      spans.append(c0)
      s0 = []
      p0 = []
      c0 = []
      stack = []
      continue

    info = line.split()
    s0.append(info[0]) # word
    p0.append(info[1]) # postag
    index = len(s0) - 1

    cinfo = re.split(r'[\(\)\*]+', info[2])
    #print info[2], cinfo
    
    # push stack
    for c in cinfo:
      if c != '':
        stack.append((c, index, -1))

    # pop stack
    for c in info[2]:
      if c == ')':
        c0.append((stack[-1][0], stack[-1][1], index))
        stack.pop(-1)    
  
  # for each line
  if len(s0) > 0:
    sentences.append(s0)
    postags.append(p0)
    spans.append(c0)

  fin.close()
  return sentences, postags, spans

if __name__ == '__main__':
  sentences, postags, spans = extract_gold_syntax_spans(sys.argv[1])
  print(sentences[0], postags[0], spans[0])
