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

from Treebank.PTB import PTBFile

def clean_file(f):
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


def write_train(ptb_loc, fold_dir):
    ptb_loc = Path(ptb_loc)
    fold_dir = Path(fold_dir)
    all_trees = []
    swbd_header_re = re.compile(r'\*x\*.+\*x\*\n')
    speaker_code_re = re.compile(r'\( \(CODE .+\n')
    for fn in fold_dir.join('mrg-train-filenames.txt').open():
        ptb_file = PTBFile(path=str(ptb_loc.join(str(fn.strip()))))
        clean_file(ptb_file)
        all_trees.extend('( ' + str(sent) + ')' for sent in ptb_file.children())
    # Reserve 1000 sentences for held out
    heldout = all_trees[:1000]
    train = all_trees[20:]
    fold_dir.join('train.mrg').open('w').write(u'\n'.join(train))
    fold_dir.join('heldout.mrg').open('w').write(u'\n'.join(heldout))


def train_parser(bllip_loc, fold_dir):
    bllip_loc = Path(bllip_loc)
    data_loc = bllip_loc.join('first-stage/DATA').join('LM')
    sh.cp('-r', str(data_loc), str(fold_dir))
    sh.trainParser('-lm', '-En',
                   str(fold_dir.join('LM')),
                   str(fold_dir.join('train.mrg')),
                   str(fold_dir.join('heldout.mrg')))

def get_nbests(raw_text):
    nbest_re = re.compile(r'\nN=\d+\t')
    for raw_nbest in nbest_re.split(raw_text):
        nbest = []
        for raw_sent in raw_nbest.split('\n')[1:]:
            raw_words = raw_sent.split()[7:]
            words = [w for (i, w) in enumerate(raw_words)[:-1] if raw_words[i + 1] == '_']
            nbest.append('<s> ' + ' '.join(words) + '</s>')
            yield '\n'.join(nbest)


def parse_test(fold_dir):
    for nbest in get_nbests(fold_dir.join('test-strings.txt').open().read()):
        print '\n'.join(nbest)
        #open('/tmp/tests.txt', 'w').write(nbest)
        #parses = sh.parseIt('-M', '-C', '-K', '-L', 'En', str(fold_dir.join('LM')),
        #           '/tmp/tests.txt').stdout
        #print parses


def main(bllip_loc, cvfolds):
    ptb_loc = '/usr/local/data/Penn3/parsed/mrg/swbd/'
    for fold_dir in Path(cvfolds):
        print fold_dir
        write_train(ptb_loc, fold_dir)
        #train_parser(bllip_loc, fold_dir)
        parse_test(fold_dir)
        break


if __name__ == '__main__':
    plac.call(main)
