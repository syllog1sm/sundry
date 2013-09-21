"""
Find the best interpolation constant for two scores. We use simple brute-force
search at a given resolution.

Input data format:

sentence1_candidate1_WER\tScore1\tScore2
sentence1_candidate2_WER\tScore1\tScore2
...
sentence1_candidate50_WER\tScore1\tScore2

sentence2_candidate1_WER\tScore1\tScore2

i.e., give three columns, for the WER, first score, second score. Columns are
tab-delimited. Leave an additional blank line between sentences.
"""
import sys
import math

def macro_score(nbests, weight):
    inv_w = 1 - weight
    # TODO: This computes a macro-averaged WER. Is this correct? Micro-average
    # seems more sensible to me
    total = 0
    for nbest in nbests:
        best = min(nbest, key=lambda c: (c[1] * weight) + (c[2] * inv_w))
        # To get a micro-average, we just need to weight this by the sentence length
        total += best[0]
    return total / len(nbests)



def steps(step_size):
    for i in range(int(1 / step_size) + 1):
        v = i / math.floor(1 / step_size)
        yield v
        if v >= 1.0:
            break


def read_nbest(nbest_str):
    candidates = []
    for line in nbest_str.strip().split('\n'):
        wer, score1, score2 = line.split()
        candidates.append((float(wer), float(score1), float(score2)))
    return candidates


def main(in_, step_size=0.05):
    # Read in the data
    nbests = [read_nbest(nbest) for nbest in in_.read().strip().split('\n\n')]
    
    # Do the arg-max. We compute a score for the nbest lists at each step-size,
    # and find the (score, step-size) pair that's highest.
    best_weight = 0
    best_score = 100000
    for w in steps(step_size):
        s = macro_score(nbests, w)
        print w, s
        if s <= best_score:
            best_score = s
            best_weight = w
    print "Best:", best_weight, best_score


if __name__ == '__main__':
    if len(sys.argv) >= 2:  
        main(sys.stdin, sys.argv[1])
    else:
        main(sys.stdin)
