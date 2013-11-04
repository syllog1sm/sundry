"""A simple implementation of a greedy transition-based parser"""
import random
from _perceptron import Perceptron
from taggers import PerceptronTagger
import os
from os import path


class Parse(object):
    def __init__(self, n):
        self.heads = [None] * (n + 1)
        self.lefts = []
        self.rights = []
        for i in range(n+1):
            self.lefts.append([0, 0])
            self.rights.append([0, 0])
        self.labels = [None] * (n + 1)

    def add(self, head, child, label=None):
        former = self.heads[child]
        if former:
            assert former < child
            popped = self.rights[former].pop()
            assert popped == child
        self.heads[child] = head
        if head > child:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)


class State(object):
    def __init__(self, n):
        self.n0 = 1
        self.n = n - 1
        self.stack = []
        self.parse = Parse(n)

    @property
    def s0(self):
        return self.stack[-1] if self.stack else 0

    def push(self):
        self.stack.append(self.n0)
        self.n0 += 1

    def pop(self):
        self.stack.pop()


SHIFT = 0; REDUCE = 1; LEFT = 2; RIGHT = 3
MOVES = [SHIFT, REDUCE, LEFT, RIGHT]


def transition(state, move):
    if move == SHIFT:
        state.push()
    elif move == REDUCE:
        if state.parse.heads[state.s0] is None and len(state.stack) >= 2:
            state.parse.add(state.stack[-2], state.stack[-1])
        state.pop()
    elif move == RIGHT:
        state.parse.add(state.s0, state.n0)
        state.push()
    elif move == LEFT:
        state.parse.add(state.n0, state.s0)
        state.pop()
    else:
        raise StandardError


def get_valid(state):
    invalid = set()
    if state.n0 >= state.n:
        invalid.add(SHIFT); invalid.add(RIGHT)
    if not state.stack:
        invalid.add(REDUCE); invalid.add(RIGHT); invalid.add(LEFT)
    if len(invalid) == len(MOVES):
        print state.n0, state.s0, state.n
        raise StandardError
    return [m for m in MOVES if m not in invalid]


def get_optimal(moves, n0, s0, n, stack, heads):
    invalid = set()
    if s0 != 0 and n0 != 0 and heads[n0] == s0:
        return [RIGHT]
    elif heads[n0] > n0 or heads[n0] in stack:
        invalid.add(RIGHT)
    if s0 != 0 and n0 != 0 and heads[s0] == n0:
        return [LEFT]
    elif len(stack) >= 2 and stack[-2] == heads[s0]:
        invalid.add(LEFT)
    if any(heads[w] == n0 or heads[n0] == w for w in stack[:-1]):
        invalid.add(SHIFT); invalid.add(RIGHT)
    if heads[s0] > n0 or any(heads[w] == s0 for w in range(n0 + 1, n)):
        invalid.add(LEFT); invalid.add(REDUCE)
    moves = [m for m in moves if m not in invalid]
    assert moves
    return moves
    

class Parser:
    def __init__(self, model_dir):
        self.model = Perceptron()
        self.tagger = PerceptronTagger(path.join(model_dir, 'tagger'))
    
    def parse(self, words, tags):
        #tags = [None] + self.tagger.tag(words[1:])
        state = State(len(words))
        while state.stack or state.n0 < state.n:
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
        while s.stack or s.n0 < s.n:
            features = extract_features(words, gold_tags, s)
            scores = self.model.score(features)
            valid_moves = get_valid(s)
            gold_moves = get_optimal(valid_moves, s.n0, s.s0, s.n, s.stack, gold.heads)
            guess = max(valid_moves, key=lambda move: scores[move])
            best = max(gold_moves, key=lambda move: scores[move])
            print guess, best, scores[guess], scores[best], len(features)
            print [(f, self.model.weights.get(f, {}).get(guess)) for f in features.keys()
                    if self.model.weights.get(f, {}).get(guess)]

            self.model.update(best, guess, features)
            #if itn >= 2 and random.random() < 0.9:
            transition(s, guess)
            #else:
            #    transition(s, best)
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
    def add(name, *values):
        if any(values):
            features[name + ' '.join(values)] = 1
    
    # Setup
    heads = state.parse.heads; lefts = state.parse.lefts; rights = state.parse.rights
    n0 = state.n0; s0 = state.s0; length = state.n
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
    Vn0L = str(min((len(lefts[n0]) - 2, 5)))
    Vs0L = str(min((len(lefts[s0]) - 2, 5)))
    Vs0R = str(min((len(rights[s0]) - 2, 5)))
    # String-distance
    Ds0n0 = str(min((n0 - s0, 5)) if s0 != 0 else 0)
    Bs0h = '1' if s0h else ''
    Bs0h2 = '1' if s0h2 else ''
    Bs0R1 = '1' if s0R1 else ''
    Bs0R2 = '1' if s0R2 else ''
    Bs0L1 = '1' if s0L1 else ''
    Bs0L2 = '1' if s0L2 else ''
    Bn0L1 = '1' if n0L1 else ''
    Bn0L2 = '1' if n0L2 else ''
    # Now start adding the features
    # Bias
    add('bias', '1')
    # Unigram, word
    add('w-n0', Wn0); add('w-n1', Wn1); add('w-n2', Wn2)
    add('w-s0', Ws0); add('w-s0h', Ws0h); add('w-s0h2', Ws0h2)
    add('w-n0.l[0]', Wn0L1); add('w-n0.l[1]', Wn0L2)
    add('w-s0.l[0]', Ws0L1); add('w-s0.l[1]', Ws0L2)
    add('w-s0.r[0]', Ws0R1); add('w-s0.r[1]', Ws0R2)
    # Unigram, tag
    add('t-n0', Tn0); add('t-n1', Tn1); add('t-n2', Tn2)
    add('t-s0', Ts0); add('t-s0h', Ts0h); add('t-s0h2', Ts0h2)
    add('t-n0.l[0]', Tn0L1); add('t-n0.l[1]', Tn0L2)
    add('t-s0.l[0]', Ts0L1); add('t-s0.l[1]', Ts0L2)
    add('t-s0.r[0]', Ts0R1); add('t-s0.r[1]', Ts0R2)
    # Unigram, word+tag
    add('wt-n0', Wn0, Tn0); add('wt-n1', Wn1, Tn1); add('wt-n2', Wn2, Tn2)
    add('wt-s0', Ws0, Ts0)
    # Bigram, word/word
    add('w-s0+w-n0', Wn0, Ws0)
    # Bigram, word+tag/word
    add('wt-s0, w-n0', Ws0, Ts0, Wn0)
    add('w-s0, wt-n0', Ws0, Wn0, Tn0)
    # Bigram, word+tag/tag
    add('wt-s0+t-n0', Ws0, Ts0, Tn0)
    add('t-s0, wt-n0', Ts0, Wn0, Tn0)
    # Bigram, word+tag/word+tag
    add('wt-s0, wt-n0', Ws0, Ts0, Wn0, Tn0)
    # Bigram, tag/tag
    add('t-s0, t-n0', Ts0, Tn0)
    add('t-n0, t-n1', Tn0, Tn1)
    # Trigram, tag/tag/tag
    add('buffer tritag', Tn0, Tn1, Tn2)
    add('s0, n0, n1 tritag', Ts0, Tn0, Tn1)
    add('s0, s0h, n0 tritag', Ts0, Ts0h, Tn0)
    add('s0, s0L, n0 tritag', Ts0, Ts0L1, Tn0)
    add('s0, s0R, n0 tritag', Ts0, Ts0R1, Tn0)
    add('s0, n0, n0L tritag', Ts0, Tn0, Tn0L1)
    add('s0 left tritag', Ts0, Ts0L1, Ts0L2)
    add('s0 right tritag', Ts0, Ts0R1, Ts0R2)
    add('n0 left tritag', Tn0, Tn0L1, Tn0L2)
    add('stack tritag', Ts0, Ts0h, Ts0h2)
    # 'Valency' is number of children attached to a word from left or right
    # Valency + word
    add('s0 right val. + word', Ws0, Vs0R)
    add('s0 left val. + word', Ws0, Vs0L)
    add('n0 left val. + word', Wn0, Vn0L)
    # Valency + tag
    add('s0 right val. + tag', Ts0, Vs0R)
    add('s0 left val. + tag', Ts0, Vs0L)
    add('n0 left val. + tag', Tn0, Vn0L)
    # 'Distance' is the number of tokens between s0 and n0
    # Distance + word
    add('dist + w-s0', Ds0n0, Ws0)
    add('dist + w-n0', Ds0n0, Wn0)
    # Distance + tag
    add('dist + t-s0', Ds0n0, Ts0)
    add('dist + t-n0', Ds0n0, Tn0)
    # Bigram, word/word + distance
    add('dist + w-s0 + w-n0', Ds0n0, Ws0, Wn0)
    # Bigram, tag/tag + distance
    add('dist + t-s0 + t-n0', Ds0n0, Ts0, Tn0)
    # "Label" features
    add('S0l', Bs0h)
    add('S0hl', Bs0h2)
    add('S0ll', Bs0L1)
    add('S0l2l', Bs0L2)
    add('S0rl', Bs0R1)
    add('S0r2l', Bs0R1)
    add('N0ll', Bn0L1)
    add('N0l2l', Bn0L2)
    add('S0w, S0rl, S0r2l', Ws0, Bs0R1, Bs0R2)
    add('S0p, S0rl, S0r2l', Ts0, Bs0R1, Bs0R2)
    add('S0w, S0ll, S0l2l', Ws0, Bs0L1, Bs0L2)
    add('S0p, S0ll, S0l2l', Ts0, Bs0L1, Bs0L2)
    add('N0w, N0ll, N0l2l', Wn0, Bn0L1, Bn0L2)
    add('N0p, N0ll, N0l2l', Tn0, Bn0L1, Bn0L2)
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
    sentences = list(read_conll(train_loc))[:10]
    parser.train(sentences, nr_iter=5)
    parser.model.save('/tmp/parser.pickle')
    c = 0
    t = 0
    for words, gold_tags, gold_parse in read_conll(heldout_loc):
        tags, heads = parser.parse(words, gold_tags)
        for i, w in list(enumerate(words))[1:-1]:
            if gold_parse.labels[i] in ('P', 'punct'):
                continue
            if heads[i] == gold_parse.heads[i]:
                c += 1
            t += 1
    print c, t, float(c)/t


if __name__ == '__main__':
    import plac
    import cProfile
    import pstats
    cProfile.runctx('plac.call(main)', globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
