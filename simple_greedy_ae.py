"""A simple implementation of a greedy transition-based parser"""
import random
from _perceptron import Perceptron
from taggers import PerceptronTagger
import os
from os import path
import time


SHIFT = 0; REDUCE = 1; LEFT = 2;
MOVES = (SHIFT, REDUCE, LEFT)



class DefaultList(list):
    """A list that returns a default value if index out of bounds."""
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default


class State(object):
    def __init__(self, n):
        self.i = 1 
        self.n = n
        self.stack = DefaultList(0)
        self.heads = [None] * (n-1)
        self.lefts = []
        self.rights = []
        for i in range(n+1):
            self.lefts.append(DefaultList(0))
            self.rights.append(DefaultList(0))
        self.stack.append(self.i)
        self.i += 1

    def transition(self, move):
        s0 = self.stack[-1]
        s1 = self.stack[-2]
        n0 = self.i
        if move == SHIFT:
            self.heads[n0] = s0
            self.rights[s0].append(n0)
            self.stack.append(n0)
            self.i += 1
        elif move == REDUCE:
            self.stack.pop()
        elif move == LEFT:
            if s1 != 0:
                self.rights[s1].pop()
            self.heads[s0] = n0
            self.lefts[n0].append(s0)
            self.stack.pop()
        else:
            raise StandardError
        if not self.stack and (self.i + 1) < self.n:
            self.stack.append(self.i)
            self.i += 1

    def context(self):
        stack = self.stack
        n0 = self.i
        s0 = stack[-1]
        s0l = self.lefts[s0]
        s0r = self.rights[s0]
        n0l = self.lefts[self.i]
        return (stack[-3], self.stack[-2], s0l[-1], s0l[-2], s0, s0r[-2], s0r[-1],
                n0l[-1], n0l[-2], n0, n0+1, n0+2)

    def oracle(self, gold):
        n0 = self.i
        s0 = self.stack[-1]
        if gold[n0] == s0:
            return [SHIFT]
        elif gold[s0] == n0:
            return [LEFT]
        invalid = set()
        if (self.i + 1) == self.n:
            invalid.add(SHIFT)
        if gold[s0] == self.heads[s0]:
            invalid.add(LEFT)
        # If there are any dependencies between n0 and the stack,
        # pushing n0 will lose them.
        for w in self.stack[:-1]:
            if gold[w] == n0 or gold[n0] == w:
                invalid.add(SHIFT)
                break
        # If there are any dependencies between s0 and the buffer, popping
        # s0 will lose them.
        for w in range(self.i+1, self.n):
            if gold[w] == s0 or gold[s0] == w:
                invalid.add(LEFT)
                invalid.add(REDUCE)
                break
        return [m for m in MOVES if m not in invalid]


class Parser:
    def __init__(self, model_dir):
        self.model = Perceptron()
        self.tagger = PerceptronTagger(path.join(model_dir, 'tagger'))
    
    def parse(self, words, tags):
        state = State(len(words))
        while state.stack or (state.i + 1) < state.n:
            features = extract_features(words, tags, state)
            scores = self.model.score(features)
            moves = MOVES if (state.i + 1) < state.n else [LEFT, REDUCE]
            guess = max(moves, key=lambda move: scores[move])
            state.transition(guess)
        return tags, state.heads

    def train_one(self, itn, words, gold_tags, gold_heads):
        s = State(len(words))
        c = 0
        while s.stack or (s.i + 1) < s.n:
            features = extract_features(words, gold_tags, s)
            scores = self.model.score(features)
            moves = MOVES if (s.i + 1) < s.n else [LEFT, REDUCE]
            gold_moves = s.oracle(gold_heads)
            guess = max(moves, key=lambda move: scores[move])
            best = max(gold_moves, key=lambda move: scores[move])
            self.model.update(best, guess, features)
            s.transition(guess)
            c += guess == best
        return c

    def train(self, sentences, nr_iter=15):
        total = 0
        for itn in range(nr_iter):
            corr = 0; total = 0
            random.shuffle(sentences)
            for words, gold_tags, gold_parse, gold_label in sentences:
                corr += self.train_one(itn, words, gold_tags, gold_parse)
                total += (len(words) - 3) * 2
            print corr, total, float(corr) / float(total) 
        self.model.average_weights()


def extract_features(words, tags, state):
    features = {}
    # Setup
    s2, s1, s0L1, s0L2, s0, s0R1, s0R2, n0L1, n0L2, n0, n1, n2 = state.context()
    # Word features for the above token indices
    Wn0 = words[n0]; Wn1 = words[n1]; Wn2 = words[n2]
    Ws0 = words[s0]; Ws1 = words[s1]; Ws2 = words[s2]
    Wn0L1 = words[n0L1]; Wn0L2 = words[n0L2]
    Ws0L1 = words[s0L1]; Ws0L2 = words[s0L2]
    Ws0R1 = words[s0R1]; Ws0R2 = words[s0R2]
    # Part-of-speech tag features
    Tn0 = tags[n0]; Tn1 = tags[n1]; Tn2 = tags[n2]
    Ts0 = tags[s0]; Ts1 = tags[s1]; Ts2 = tags[s2]
    Tn0L1 = tags[n0L1]; Tn0L2 = tags[n0L2]
    Ts0L1 = tags[s0L1]; Ts0L2 = tags[s0L2]
    Ts0R1 = tags[s0R1]; Ts0R2 = tags[s0R2]
    # Cap numeric features at 5 
    # Valency (number of children) features
    Vn0L = len(state.lefts[n0])
    Vs0L = len(state.lefts[s0])
    Vs0R = len(state.rights[s0])
    # String-distance
    Ds0n0 = min((n0 - s0, 5)) if s0 != 0 else 0

    features['bias'] = 1
    w = (Wn0, Wn1, Wn2, Ws0, Ws1, Ws2, Wn0L1, Wn0L2, Ws0L1, Ws0L2, Ws0R1, Ws0R2)
    t = (Tn0, Tn1, Tn2, Ts0, Ts1, Ts2, Tn0L1, Tn0L2, Ts0L1, Ts0L2, Ts0R1, Ts0R2)
    for code, templates in zip(('w', 't'), (w, t)):
        for i, value in enumerate(templates):
            if value:
                features['%s%d %s' % (code, i, value)] = 1

    wt = ((Wn0, Tn0), (Wn1, Tn1), (Wn2, Tn2), (Ws0, Ts0))
    for i, (word, tag) in enumerate(wt):
        if word or tag:
            features['wt-%d %s %s' % (i, word, tag)] = 1
    features['ww %s %s' % (Ws0, Wn0)] = 1
    features['wn0tn0-ws0 %s/%s %s' % (Wn0, Tn0, Ws0)] = 1
    features['wn0tn0-ts0 %s/%s %s' % (Wn0, Tn0, Ts0)] = 1
    features['ws0ts0-wn0 %s/%s %s' % (Ws0, Ts0, Wn0)] = 1
    features['ws0-ts0 tn0 %s/%s %s' % (Ws0, Ts0, Tn0)] = 1
    features['wt-wt %s/%s %s/%s' % (Ws0, Ts0, Wn0, Tn0)] = 1
    features['tt s0=%s n0=%s' % (Ts0, Tn0)] = 1
    features['tt n0=%s n1=%s' % (Tn0, Tn1)] = 1

    trigrams = ((Tn0, Tn1, Tn2), (Ts0, Tn0, Tn1), (Ts0, Ts1, Tn0), 
                (Ts0, Ts0L1, Tn0), (Ts0, Ts0R1, Tn0), (Ts0, Tn0, Tn0L1),
                (Ts0, Ts0L1, Ts0L2), (Ts0, Ts0R1, Ts0R2), (Tn0, Tn0L1, Tn0L2),
                (Ts0, Ts1, Ts1))
    for i, (t1, t2, t3) in enumerate(trigrams):
        if t1 or t2 or t3:
            features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1
    vw = ((Ws0, Vs0R), (Ws0, Vs0L), (Wn0, Vn0L))
    vt = ((Ts0, Vs0R), (Ts0, Vs0L), (Tn0, Vn0L))
    d = ((Ws0, Ds0n0), (Wn0, Ds0n0), (Ts0, Ds0n0), (Tn0, Ds0n0),
         ('t' + Tn0+Ts0, Ds0n0), ('w' + Wn0+Ws0, Ds0n0))
    for i, (w_t, v_d) in enumerate(vw + vt + d):
        if w_t or v_d:
            features['val/d-%d %s %d' % (i, w_t, v_d)] = 1
    return features


def read_conll(loc):
    for sent_str in open(loc).read().strip().split('\n\n'):
        lines = [line.split() for line in sent_str.split('\n')]
        words = DefaultList(''); tags = DefaultList('')
        words.append('<start>'); tags.append('<start>')
        heads = [None]; labels = [None]
        for i, (word, pos, head, label) in enumerate(lines):
            if '-' in word and word[0] != '-':
                word = '!HYPHEN'
            elif word.isdigit() and len(word) == 4:
                word = '!YEAR'
            elif word[0].isdigit():
                word = '!DIGITS'
            else:
                word = word.lower()
            words.append(intern(word))
            tags.append(intern(pos))
            heads.append(int(head) + 1 if head != '-1' else len(lines) + 1)
            labels.append(label)
        words.append('ROOT'); tags.append('ROOT')
        heads.append(None); labels.append(None)
        yield words, tags, heads, labels

        
def main(model_dir, train_loc, heldout_loc):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    parser = Parser(model_dir)
    sentences = list(read_conll(train_loc))[:5000]
    parser.train(sentences, nr_iter=15)
    parser.model.save('/tmp/parser.pickle')
    c = 0
    t = 0
    t1 = time.time()
    for words, gold_tags, gold_heads, gold_labels in read_conll(heldout_loc):
        tags, heads = parser.parse(words, gold_tags)
        for i, w in list(enumerate(words))[1:-1]:
            if gold_labels[i] in ('P', 'punct'):
                continue
            if heads[i] == gold_heads[i]:
                c += 1
            t += 1
    t2 = time.time()
    print 'Parsing took %0.3f ms' % ((t2-t1)*1000.0)
    print c, t, float(c)/t


if __name__ == '__main__':
    import sys
    import cProfile
    import pstats
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    #cProfile.runctx('main(sys.argv[1], sys.argv[2], sys.argv[3])', globals(),
    #$                locals(), "Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()
