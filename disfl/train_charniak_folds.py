"""Train Charniak Parsers for the different folds, and output scores for
the test sentences.

Each fold has the files:
    - dps-train-filenames.txt
    - mrg-test-filenames.txt
    - mrg-train-filenames.txt
    - test-strings.txt

We need to concatenate the sentences in the files referred to by mrg-train-filenames,
pre-process the strings in test-strings, pass the text to the trained parsers,
and write in the scores."""
from pathlib import Path
import plac
import sh
import re
import math

from Treebank.PTB import PTBFile

class NBest(list):
    """An nbest list of candidates."""
    def __init__(self, nbest_strs):
        list.__init__(self)
        lines = nbest_strs.split('\n')
        self.gold = lines.pop(0)
        self.n = len(lines)
        self.extend(Candidate(line) for line in lines)

    def to_str(self):
        lines = [u'N=%d\t%s' % (self.n, self.gold)]
        for cand in self:
            lines.append(cand.to_str)
        return u'\n'.join(lines)


class Candidate(object):
    """A candidate disfluency analysis from an nbest list"""
    def __init__(self, i, raw_str):
        self.i = i
        pieces = raw_str.split()
        self.scores = [float(s) for s in pieces[:7]]
        self.cand_str = ' '.join(pieces[7:])
        self.words = [w for (i, w) in enumerate(pieces)[:-1]
                      if raw_words[i + 1] == '_']
        self.to_parse = '<s> ' + ' '.join(words) + ' </s>'
        self.parse = None

    def to_str(self):
        pieces = [str(score) for score in self.scores]
        pieces.append(self.cand_str)
        return '\n'.join(pieces)


def write_train(ptb_loc, fold_dir):
    ptb_loc = Path(ptb_loc)
    fold_dir = Path(fold_dir)
    all_trees = []
    swbd_header_re = re.compile(r'\*x\*.+\*x\*\n')
    speaker_code_re = re.compile(r'\( \(CODE .+\n')
    for fn in fold_dir.join('mrg-train-filenames.txt').open():
        ptb_file = PTBFile(path=str(ptb_loc.join(str(fn.strip()))))
        _clean_mrg_file(ptb_file)
        all_trees.extend('( ' + str(sent) + ')' for sent in ptb_file.children())
    # Reserve 1000 sentences for held out
    heldout = all_trees[:1000]
    train = all_trees[1000:]
    fold_dir.join('train.mrg').open('w').write(u'\n'.join(train))
    fold_dir.join('heldout.mrg').open('w').write(u'\n'.join(heldout))


def _clean_mrg_file(f):
    to_prune = set(['EDITED', 'CODE', 'RM', 'RS', 'IP', '-DFL-'])
    for sent in f.children():
        for node in sent.depthList():
            if node.label in to_prune:
                node.prune()
            elif node.isLeaf() and node.isPunct():
                node.prune()
            elif node.isLeaf() and node.text[-1] == '-':
                node.prune()
            elif node.label == 'TYPO':
                parent = node.parent()
                node.prune()
                for child in node.children():
                    child.reattach(parent)
        for node in sent.depthList():
            words = [w for w in node.listWords()]
            if not words:
                node.prune()
        words = [w for w in sent.listWords() if not w.isTrace()]
        if not words:
            f.detachChild(sent)


def train_parser(bllip_loc, fold_dir):
    bllip_loc = Path(bllip_loc)
    data_loc = bllip_loc.join('first-stage/DATA').join('LM')
    sh.cp('-r', str(data_loc), str(fold_dir))
    sh.trainParser('-lm', '-En',
                   str(fold_dir.join('LM')),
                   str(fold_dir.join('train.mrg')),
                   str(fold_dir.join('heldout.mrg')))


def parse_test(fold_dir):
    nbest_re = re.compile(r'\nN=\d+\t')
    raw_text = fold_dir.join('test-strings.txt').open().read()
    nbests = [NBest(raw) for raw in nbest_re.split(raw_text)]
    strings = []
    cand_dict = {}
    for nbest in nbests:
        for candidate in nbest:
            strings.append(candidate.to_parse)
            cand_dict[candidate.to_parse] = candidate
    _add_parse_scores(strings, cand_dict)
    out_file = fold_dir.join('scored-test.txt').open('w')
    for nbest in nbests:
        out_file.write(unicode(nbest.to_str()))
        out_file.write(u'\n')
    out_file.close()


def _add_parse_scores(strings, cand_dict):
    open('/tmp/to_parse.txt', 'w').write('\n'.join(strings))
    parses = sh.parseIt('-M', '-C', '-K', str(fold_dir.join('LM')),
                        '/tmp/tests.txt').stdout
    for parse_and_scores in parses.split('\n\n'):
        scores, parse = parse_and_score.split('\n')
        input_str = get_str(parse)
        candidate = cand_dict[input_str]
        candidate.parse = parse
        candidate.scores.extend(math.exp(float(s)) for s in scores.split())


def main(bllip_loc, cvfolds):
    ptb_loc = '/usr/local/data/Penn3/parsed/mrg/swbd/'
    for fold_dir in Path(cvfolds):
        print fold_dir
        write_train(ptb_loc, fold_dir)
        train_parser(bllip_loc, fold_dir)
        parse_test(fold_dir)
        break


if __name__ == '__main__':
    plac.call(main)
