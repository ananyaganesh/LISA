Usage (Python2):

1) Analysis for SRL constraint violation, syntactic agreement, long-range dependencies:
 python full_analysis.py pradhan2.devel

2) Analysis for error-breakdown and confusion matrix:
 python srl_analysis.py pradhan2.devel

pradhan2.devel is the CoNLL formatted file same as the one accepted by the official conlleval script. (from Pradhan et al., downloaded from the CoNLL05 website, for demonstration purpose ...)

Quirks:
1) The F1 (in error-breakdown, etc.) will be slightly different from official F1 (by about 0.1%, because of different handling of C-X args, which I decided to ignore for error analysis purpose ...)
2) The scripts works for CoNLL05 dev set (which depends on the hardcoded core Arg set and the gold CoNLL05 SRL and syntax data). If you would like to do analysis on other datasets, such as the CoNLL-2012, you probably need to change at least the core Arg set, and the path to the gold SRL/syntax files.

