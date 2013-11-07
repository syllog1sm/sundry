"""A simple implementation of a greedy transition-based parser"""
import random
from _perceptron import Perceptron
from taggers import PerceptronTagger
import os
from os import path
import time


class Parse(object):
    def __init__(self, n):
        self.heads = [None] * (n + 1)
        self.lefts = []
        self.rights = []
        # Pad these, for easy look up
        for i in range(n+1):
            self.lefts.append([0, 0])
            self.rights.append([0, 0])
        self.labels = [None] * (n + 1)

    def add(self, head, child, label=None):
        self.heads[child] = head
        if head > child:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)


class State(object):
    def __init__(self, n):
        self.i = 2 
        self.n = n - 1
        self.stack = [1]
        self.parse = Parse(n)

    @property
    def s0(self):
        return self.stack[-1] if self.stack else 0

    def push(self):
        self.stack.append(self.i)
        self.i += 1

    def pop(self):
        self.stack.pop()
        if not self.stack and (self.i < self.n):
            self.push()

    def add(self, head, child):
        self.heads[child] = head
        if head > child:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)


SHIFT = 0; REDUCE = 1; LEFT = 2;
MOVES = [SHIFT, REDUCE, LEFT]


def transition(state, move):
    if move == SHIFT:
        state.parse.add(state.s0, state.i)
        state.push()
    elif move == REDUCE:
        state.pop()
    elif move == LEFT:
        if state.parse.heads[state.s0]:
            state.parse.rights[state.stack[-2]].pop()
        state.parse.add(state.i, state.s0)
        state.pop()
    else:
        raise StandardError


def get_optimal(valid, state, gold):
    n0 = state.i
    s0 = state.s0
    if gold[n0] == s0:
        return [SHIFT]
    elif gold[s0] == n0:
        return [LEFT]
    invalid = set()
    if gold[s0] == state.parse.heads[s0]:
        invalid.add(LEFT)
    for w in state.stack:
        if gold[w] == n0 or gold[n0] == w:
            invalid.add(SHIFT)
            break
    for w in range(state.i, state.n+1):
        if gold[w] == s0 or gold[s0] == w:
            invalid.add(LEFT)
            invalid.add(REDUCE)
    moves = [m for m in valid if m not in invalid]
    assert moves
    return moves

def get_valid(state):
    assert state.stack
    valid = []
    if state.i < state.n:
        valid.append(SHIFT)
    valid.append(REDUCE)
    valid.append(LEFT)
    return valid

class Parser:
    def __init__(self, model_dir):
        self.model = Perceptron()
        self.tagger = PerceptronTagger(path.join(model_dir, 'tagger'))
    
    def parse(self, words, tags):
        #tags = [None] + self.tagger.tag(words[1:])
        state = State(len(words))
        while state.stack or state.i < state.n:
            features = extract_features(words, tags, state)
            scores = self.model.score(features)
            valid_moves = get_valid(state)
            guess = max(valid_moves, key=lambda move: scores[move])
            transition(state, guess)
        return tags, state.parse.heads

    def train_one(self, itn, words, gold_tags, gold):
        #tags = [None] + self.tagger.tag(words[1:])
        s = State(len(words))
        c = 0
        i = 0
        #for c, h in enumerate(gold.heads):
        #    print c, words[c], words[h] if h is not None else '-None-'
        move_strs = ['S', 'D', 'L']
        while s.stack or s.i < s.n:
            features = extract_features(words, gold_tags, s)
            scores = self.model.score(features)
            valid_moves = get_valid(s)
            gold_moves = get_optimal(valid_moves, s, gold.heads)
            guess = max(valid_moves, key=lambda move: scores[move])
            best = max(gold_moves, key=lambda move: scores[move])
            label = '(' + move_strs[guess] + ', ' + move_strs[best] + ')'
            #print label, ' '.join(words[s] for s in s.stack), '|', words[s.n0]
            #print s.s0, s.n0
            #print s.parse.heads[s.s0], gold.heads[s.s0]
            #print [w for w, h in enumerate(gold.heads) if h == s.s0]
            i += 1
            self.model.update(best, guess, features)
            transition(s, guess)
            c += guess == best
        return c

    def train(self, sentences, nr_iter=15):
        #self.tagger.start_training([(words, tags) for (words, tags, _) in sentences])
        total = 0
        for itn in range(nr_iter):
            corr = 0; total = 0
            #random.shuffle(sentences)
            for words, gold_tags, gold_parse in sentences:
                corr += self.train_one(itn, words, gold_tags, gold_parse)
                #self.tagger.train_one(words[1:], gold_tags[1:])
                total += (len(words) - 2) * 2
            print corr, total, float(corr) / float(total) 
        #self.tagger.end_training()
        self.model.average_weights()


def extract_features(words, tags, state):
    features = {}
    # Setup
    heads = state.parse.heads; lefts = state.parse.lefts; rights = state.parse.rights
    n0 = state.i; s0 = state.s0; length = state.n
    # We need to pick out the context tokens we'll be dealing with, and also
    # the pieces we'll be constructing features from. We mustn't get these
    # confused!!
    # To help, there's a convention: properties that can be part of features start
    # with upper-case; token indices start with lower-case.
    #
    # An argument of add() that starts lower-case is a bug!!
    # An array index that starts upper-case is a bug!!
    #
    n1 = (n0 + 1) if (n0 + 1) < length else 0 
    n2 = (n0 + 2) if (n0 + 2) < length else 0
    s0h = heads[s0] if heads[s0] is not None else 0
    s0h2 = heads[s0h] if heads[s0h] is not None else 0
    # Tokens of s0 and n0's subtrees.
    n0L1 = lefts[n0][-1]; n0L2 = lefts[n0][-2]
    s0L1 = lefts[s0][-1]; s0L2 = lefts[s0][-2]
    s0R1 = rights[s0][-1]; s0R2 = rights[s0][-2]
    # Word features for the above token indices
    Wn0 = words[n0]; Wn1 = words[n1]; Wn2 = words[n2]
    Ws0 = words[s0]; Ws0h = words[s0h]; Ws0h2 = words[s0h2]
    Wn0L1 = words[n0L1]; Wn0L2 = words[n0L2]
    Ws0L1 = words[s0L1]; Ws0L2 = words[s0L2]
    Ws0R1 = words[s0R1]; Ws0R2 = words[s0R2]
    # Part-of-speech tag features
    Tn0 = tags[n0]; Tn1 = tags[n1]; Tn2 = tags[n2]
    Ts0 = tags[s0]; Ts0h = tags[s0h]; Ts0h2 = tags[s0h2]
    Tn0L1 = tags[n0L1]; Tn0L2 = tags[n0L2]
    Ts0L1 = tags[s0L1]; Ts0L2 = tags[s0L2]
    Ts0R1 = tags[s0R1]; Ts0R2 = tags[s0R2]
    # Cap numeric features at 5 
    # Valency (number of children) features
    Vn0L = len(lefts[n0]) - 2
    Vs0L = len(lefts[s0]) - 2
    Vs0R = len(rights[s0]) - 2
    # String-distance
    Ds0n0 = min((n0 - s0, 5)) if s0 != 0 else 0

    features['bias'] = 1
    w = (Wn0, Wn1, Wn2, Ws0, Ws0h, Ws0h2, Wn0L1, Wn0L2, Ws0L1, Ws0L2, Ws0R1, Ws0R2)
    t = (Tn0, Tn1, Tn2, Ts0, Ts0h, Ts0h2, Tn0L1, Tn0L2, Ts0L1, Ts0L2, Ts0R1, Ts0R2)
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

    trigrams = ((Tn0, Tn1, Tn2), (Ts0, Tn0, Tn1), (Ts0, Ts0h, Tn0), 
                (Ts0, Ts0L1, Tn0), (Ts0, Ts0R1, Tn0), (Ts0, Tn0, Tn0L1),
                (Ts0, Ts0L1, Ts0L2), (Ts0, Ts0R1, Ts0R2), (Tn0, Tn0L1, Tn0L2),
                (Ts0, Ts0h, Ts0h2))
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
        words = ['']; tags = ['']
        parse = Parse(len(lines) + 1)
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
            parse.add(int(head) + 1 if head != '-1' else len(lines) + 1, i + 1,
                      label=label)
        words.append('ROOT'); tags.append('ROOT')
        assert not [w for w in parse.heads[1:-1] if w is None]
        yield words, tags, parse

        
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
    for words, gold_tags, gold_parse in read_conll(heldout_loc):
        tags, heads = parser.parse(words, gold_tags)
        for i, w in list(enumerate(words))[1:]:
            if gold_parse.labels[i] in ('P', 'punct'):
                continue
            if heads[i] == gold_parse.heads[i]:
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
