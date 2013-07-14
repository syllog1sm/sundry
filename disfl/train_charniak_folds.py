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

def write_train(ptb_loc, fold_dir):
    ptb_loc = Path(ptb_loc)
    fold_dir = Path(fold_dir)
    text = []
    for fn in fold_dir.join('mrg-train-filenames.txt').open():
        text.append(ptb_loc.join(str(fn.strip())).open().read().strip())
    fold_dir.join('train.mrg').open('w').write(u'\n'.join(text))


def main(cvfolds):
    ptb_loc = '/usr/local/data/Penn3/parsed/mrg/swbd/'
    for fold_dir in Path(cvfolds):
        write_train(ptb_loc, fold_dir)
        #train_parser(fold_dir)


if __name__ == '__main__':
    plac.call(main)
