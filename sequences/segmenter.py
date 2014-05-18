"""Wrap Wapiti for segmentation"""

from pathlib import Path
import fabric.api


#EXT_DIR = str(Path(__file__).parent().parent().join('ext'))

LOCAL_DIR = str(Path(__file__).parent())


def train(train_seqs, **kwargs):
    train_loc = '/tmp/train.crf'
    open(train_loc, 'w').write(_seqs_to_str(train_seqs))
    args = '-T crf -a l-bfgs'
    with fabric.api.lcd(LOCAL_DIR):
        fabric.api.local(
            './wapiti train -p {pattern} {train} {model} {args}'.format(
                train=train_loc, pattern='segment.pat', model='model', args=args
            )
        )
        

def label(input_seqs):
    input_loc = '/tmp/heldout.crf'
    open(input_loc, 'w').write(_seqs_to_str(input_seqs))
    output_loc = '/tmp/output.crf'
    with fabric.api.lcd(LOCAL_DIR):
        fabric.api.local(
            './wapiti label -c -m {model} {input} {output}'.format(
                model='model', input=input_loc, output=output_loc
            )
        )
    return open(output_loc).read()

def _read_conll(loc):
    seqs = []
    for sent_str in open(loc).read().strip().split('\n\n'):
        seq = []
        last_id = None
        for tok_str in sent_str.split('\n'):
            pieces = tok_str.split()
            word_id = int(pieces[0])
            word = pieces[1]
            pos = pieces[3]
            if last_id is not None and word_id <= last_id:
                sbd = 'T'
            else:
                sbd = 'F'
            seq.append((word, pos, sbd))
            last_id = word_id
        seqs.append(seq)
    return seqs

def _seqs_to_str(seqs):
    lines = []
    for seq in seqs:
        lines.extend('%s\t%s\t%s' % (word, pos, sbd) for word, pos, sbd in seq)
        lines.append('')
    return '\n'.join(lines)
            

def main(mode, data_loc):
    if mode == 'train':
        train(_read_conll(data_loc))
    elif mode == 'label':
        label(_read_conll(data_loc))
    elif mode == 'print':
        print _seqs_to_str(_read_conll(data_loc))t 


if __name__ == '__main__':
    import plac
    plac.call(main)
