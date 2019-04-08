import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(outfile, 'w') as out:
	with open(infile) as inf:
		for line in inf:

