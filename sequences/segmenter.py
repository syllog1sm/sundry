"""Wrap Wapiti for segmentation"""

from pathlib import Path
import fabric.api
import plac


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
        print '# ',
        fabric.api.local(
            './wapiti label -c -m {model} {input} {output}'.format(
                model='model', input=input_loc, output=output_loc
            )
        )
    return _to_pos(open(output_loc).read())

def _read_data(conll_loc, pos_loc, gold_pos=False):
    print 'Gold POS', gold_pos
    seqs = []
    pos_file = open(pos_loc)
    for conll_sent in open(conll_loc).read().strip().split('\n\n'):
        seq = []
        last_id = None
        try:
            pos_line = pos_file.next()
        except StopIteration:
            break
        pos_toks = [t.rsplit('/', 1) for t in pos_line.strip().split()]
        for i, tok_str in enumerate(conll_sent.strip().split('\n')):
            pieces = tok_str.split()
            word_id = int(pieces[0])
            word = pieces[1]
            assert pos_toks[i][0] == word
            pos = pos_toks[i][1] if not gold_pos else pieces[4]
            sbd = 'T' if last_id is not None and word_id <= last_id else 'F'
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

def _to_pos(crf_str):
    sents = []
    for sent_str in crf_str.strip().split('\n\n'):
        sent = []
        for tok_str in sent_str.split('\n'):
            word, pos, g_sbd, t_sbd = tok_str.split()
            if t_sbd == 'T':
                if sent:
                    sents.append(' '.join(sent))
                sent = []
            sent.append('%s/%s' % (word, pos))
        if sent:
            sents.append(' '.join(sent))
    return '\n'.join(sents)


@plac.annotations(
    gold_pos=("Use gold POS tags", "flag", "p", bool)
)
def main(mode, conll_loc, pos_loc, gold_pos=False):
    if mode == 'train':
        train(_read_data(conll_loc, pos_loc, gold_pos))
    elif mode == 'label':
        print label(_read_data(conll_loc, pos_loc, gold_pos))
    elif mode == 'print':
        print _seqs_to_str(_read_data(conll_loc, pos_loc, gold_pos))


if __name__ == '__main__':
    import plac
    plac.call(main)
