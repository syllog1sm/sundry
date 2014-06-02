import plac
import re
from pathlib import *
from collections import defaultdict
import gzip
import sh

def get_dps_sides(tb_dir):
    """Get sentences from a conversation side from the DPS files"""
    dysfl_dir = tb_dir.join('dysfl').join('dps').join('swbd')
    header_re = re.compile(r'=====+')
    speaker_re = re.compile(r'(Speaker)(\w)')
    turn_re = re.compile(r'\n *\n')
    sides = defaultdict(list)
    for subdir in ['2', '3', '4']:
        for path in dysfl_dir.join(subdir):
            if not path.parts[-1].endswith('dps'):
                continue
            name = path.parts[-1][:-4]
            text = path.open().read()
            header, text = header_re.split(text)
            for turn in turn_re.split(text.strip()):
                lines = turn.strip().split('\n')
                speaker = speaker_re.match(lines.pop(0)).group(2)
                sents = [get_words(line) for line in lines]
                sents = [sent for sent in sents if sent]
                #sides[name + speaker].extend(sents)
                unsegmented = ' '.join(sents)
                if unsegmented:
                    sides[name + speaker].append(unsegmented)
    return sides

def get_words(line):
    excluded_tags = set([',', '.'])
    pieces = line.split()
    words = []
    for piece in pieces:
        if '/' not in piece:
            continue
        word, pos = piece.rsplit('/', 1)
        if pos in excluded_tags:
            continue
        if word.endswith('-'):
            continue
        word = word.lower().replace('-', '')
        if words and (word.startswith("'") or word == "n't"):
            words[-1] += word
        else:
            words.append(word)
    return ' '.join(words)


def sort_into_sides(in_dir):
    """Get all files that refer to a single conversation side.
    Return the conversation name, and both sides."""
    by_side = defaultdict(lambda: defaultdict(list))
    in_dir = Path(in_dir)
    for path in sorted(in_dir):
        name, ms98, _, num = str(path.parts[-1]).split('-')
        speaker = name[-1]
        name = name[:-1]
        by_side[name][speaker].append(str(path))
    for name, sides in sorted(by_side.items()):
        yield name, sides['A'], sides['B']


def merge_sausages(name, side_files):
    n_align = 0
    merged = []
    for path in side_files:
        lines = [line for line in gzip.open(path).read().split('\n') if line.strip()]
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)
        for line in lines:
            pieces = line.split()
            pieces.pop(1)
            pieces.insert(1, str(n_align))
            merged.append(' '.join(pieces))
            n_align += 1
    header = ['name %s' % name,
              'numaligns %d' % n_align,
              'posterior 1']
    return u'\n'.join(header + merged)


def choose_sides(sausage_a, sausage_b, ref_a, ref_b):
    # Sometimes the recogniser output has the wrong speaker label. Pick the reference
    # by choosing the pairing that minimises the WER.
    a_to_a, aa_wer = align_to_ref(sausage_a, ref_a)
    a_to_b, ab_wer = align_to_ref(sausage_a, ref_b)
    b_to_b, bb_wer = align_to_ref(sausage_b, ref_b)
    b_to_a, ba_wer = align_to_ref(sausage_b, ref_a)
    if (aa_wer + bb_wer) < (ab_wer + ba_wer):
        return ((a_to_a, ref_a, 'A'), (b_to_b, ref_b, 'B'))
    else:
        return ((a_to_b, ref_b, 'B'), (b_to_a, ref_a, 'A'))

def align_to_ref(sausage, refs):
    tmp_saus = '/tmp/sausage'
    tmp_ref = '/tmp/ref'
    tmp_align = '/tmp/align'
    open(tmp_saus, 'w').write(sausage)
    open(tmp_ref, 'w').write(' '.join(refs))
    result = sh.lattice_tool('-in-lattice', tmp_saus, '-read-mesh', '-write-mesh',
                             tmp_align, '-ref-file', tmp_ref)
    pieces = result.split()
    wer = pieces[7]
    return open(tmp_align).read(), int(wer)



def cut_sausages(aligned, sbd):
    sausage_lines = aligned.strip().split('\n')
    idx = 0
    sausage_lines.pop(0)
    sausage_lines.pop(0)
    sausage_lines.pop(0)
    new_lines = []
    sent_id = 0
    sent = sbd.pop(0).split()
    orig_sent = list(sent)
    while sausage_lines:
        align_line = sausage_lines.pop(0)
        ref_line = sausage_lines.pop(0)
        assert align_line.startswith('align')
        assert ref_line.startswith('reference')
        ref_pieces = ref_line.strip().split()
        new_lines.append(align_line)
        new_lines.append(ref_line)
        if ref_pieces[-1] != '*DELETE*':
            sbd_word = sent.pop(0)
            assert sbd_word == ref_pieces[-1], '%s vs %s' % (ref_pieces[-1], sbd_word)
            if not sent:
                yield sent_id, new_lines, orig_sent
                new_lines = []
                sent_id += 1
                if not sbd and not sausage_lines:
                    break
                elif not sbd:
                    if sausage_lines[-1].split()[2] == '</s>':
                        sausage_lines.pop(0)
                        sausage_lines.pop(0)
                else:
                    sent = sbd.pop(0).split()
                    orig_sent = list(sent)

def sausage_to_nbest(name, sent_id, lines, out_dir, ref):
    out_dir = Path(out_dir)
    sausage_loc = out_dir.join('%s~%s' % (name, sent_id))
    with sausage_loc.open('w') as out_file:
        out_file.write(u'name %s~%s\n' % (name, sent_id))
        out_file.write(u'numaligns %d\n' % (len(lines) / 2))
        out_file.write(u'posterior 1\n')
        i = 0
        while lines:
            align_line = lines.pop(0)
            ref_line = lines.pop(0)
            pieces = align_line.split()
            out_file.write(u'align %d %s\n' % (i, ' '.join(pieces[2:])))
            pieces = ref_line.split()
            out_file.write(u'reference %d %s\n' % (i, pieces[-1]))
            i += 1
    ref_loc = out_dir.join('refs').join('%s~%s' % (name, sent_id))
    ref_loc.open('w').write(u'%s~%s %s\n' % (name, sent_id, ref))
    sh.lattice_tool('-read-mesh', '-in-lattice', sausage_loc,
                    '-out-nbest-dir', str(out_dir.join('unscored')),
                    '-out-nbest-dir-xscore1', out_dir.join('scores'),
                    '-ref-list', ref_loc, 
                    '-nbest-decode', 100)
    err_lines = sh.nbest_lattice('-nbest', out_dir.join('unscored').join('%s~%s.gz' % (name, sent_id)),
                     '-reference', '<s> %s </s>' % ref,
                     '-dump-errors').strip()
    nbest = gzip.open(str(out_dir.join('unscored').join('%s~%s.gz' % (name, sent_id)))).read()
    scores = gzip.open(str(out_dir.join('scores').join('%s~%s.gz' % (name, sent_id)))).read()
    output = [ref]
    n = len(ref.split())
    best_err = n
    for sent, score, err_line in zip(nbest.split('\n'), scores.split('\n'), err_lines.split('\n')):
        pieces = sent.split()
        words = pieces[2:]
        words.insert(0, score)
        wer = err_line.split()[0]
        words.insert(0, wer)
        output.append(u' '.join(words))
        if int(wer) < best_err:
            best_err = int(wer)
    out_loc = out_dir.join('nbest').join('%s~%s.nbest' % (name, sent_id))
    out_loc.open('w').write(u'\n'.join(output))
    return best_err, n


def main(sausage_dir, ptb_dir, out_dir):
    """
    Take confusion networks,
    1) merge all from the same conversation side,
    2) use SRI-LM kit to align with reference transcripts,
    3) cut into per-sentence confusion networks,
    4) then convert the per-sentence confusion nets into N-best lists.
    5) Output the oracle WER rate for the N-best list.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    for subdir in ('nbest', 'scores', 'unscored', 'refs'):
        if not out_dir.join(subdir).exists():
            out_dir.join(subdir).mkdir()
    reference = get_dps_sides(Path(ptb_dir))
    n = 0
    wer = 0
    for name, side_files_a, side_files_b in sort_into_sides(sausage_dir):
        # Ignore conversations where we don't have both sides
        if not side_files_a or not side_files_b:
            print name, "Missing side!"
            continue
        print name
        single_side_a = merge_sausages(name, side_files_a)
        single_side_b = merge_sausages(name, side_files_b)
        for side_sausage, side_ref, speaker in choose_sides(single_side_a, single_side_b,
                                                   reference[name + 'A'],
                                                   reference[name + 'B']):
            for id_, sent_sausage, ref in cut_sausages(side_sausage, side_ref):
                this_wer, this_n = sausage_to_nbest(name + speaker, id_, sent_sausage,
                                                    out_dir, u' '.join(ref))
                n += this_n
                wer += this_wer
        print wer, n, float(wer) / n
    print wer, n, float(wer) / n

if __name__ == '__main__':
    plac.call(main)
