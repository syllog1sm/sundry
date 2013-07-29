import plac
from pathlib import Path

from Treebank.PTB import PTBFile


def mark_disfl(sent):
    disfl = []
    repairs = []
    edit_depth = 0
    saw_ip = False
    for token in sent.listWords():
        if token.label == '-DFL-' and token.text == r'\[':
            edit_depth += 1
        elif token.label == '-DFL-' and token.text == r'\]' and not saw_ip and disfl:
            edit_depth -= 1
            # Assume prev token is actually repair, not reparandum
            # This should only effect 3 cases
            disfl.pop()
        elif token.label == '-DFL-' and token.text == r'\+':
            edit_depth -= 1
            saw_ip = True
        elif edit_depth >= 1:
            disfl.append(token)
    disfl = set()
    for node in sent.depthList():
        if node.label == 'EDITED':
            disfl.update(node.listWords())
    return disfl


def get_dps_string(sent):
    tokens = []
    edit_starts = set()
    edit_ends = set()
    disfl = set()
    # TODO: Comment this out to turn off marking words under EDITED as disfluent
    for node in sent.depthList():
        if node.label == 'EDITED':
            disfl.update(node.listWords())
            words = [w for w in node.listWords() if not w.isTrace() and not w.isPunct()
                     and w.label != '-DFL-']
            if words:
                edit_starts.add(words[0])
                edit_ends.add(words[-1])
    depth = 0
    edit_depth = 0
    for word in sent.listWords():
        if word.label == '-DFL-' and word.text == r'\[':
            depth += 1
            tokens.append('[')
        elif word.label == '-DFL-' and word.text == r'\]':
            if depth:
                depth -= 1
            tokens.append(']')
        elif word.label == '-DFL-' and word.text == r'\+':
            tokens.append('+')
        elif word.label == '-DFL-' and word.text[0] == 'E':
            tokens.append(word.text)
        elif word in edit_starts and depth == 0:
            tokens.append('[')
            tokens.append('%s/%s' % (word.text, word.label))
            disfl.add(word)
            if word in edit_ends:
                tokens.append(']')
            else:
                edit_depth += 1
        elif word in edit_ends and edit_depth:
            tokens.append('%s/%s' % (word.text, word.label))
            tokens.append(']')
            edit_depth -= 1
        elif word.label == 'UH':
            if word.text.lower() in set(['uh', 'um', 'uh-huh']):
                tokens.append('{F')
            else:
                tokens.append('{D')
            tokens.append('%s/%s' % (word.text, word.label))
            tokens.append('}')
        elif not word.isTrace():
            tokens.append('%s/%s' % (word.text, word.label))
            if depth:
                disfl.add(word)
    return ' '.join(tokens), disfl
        

def clean_sent(sent, disfl):
    to_prune = set(['EDITED', 'CODE', 'RM', 'RS', 'IP', '-DFL-', 'UH'])
    orig_str = str(sent)
    for node in sent.depthList():
        if node.label in to_prune:
            node.prune()
        elif node.isLeaf() and node in disfl:
            node.prune()
        elif node.isLeaf() and node.isPunct():
            node.prune()
        elif node.isLeaf() and node.isTrace():
            node.prune()
        elif node.isLeaf() and node.text.endswith('-'):
            node.prune()
    for node in sent.depthList():
        if node.label == 'TYPO':
            node.delete()
    for node in sent.depthList():
        words = [w for w in node.listWords()]
        if not words:
            node.prune()
        # Correct error
        if not node.isLeaf() and node.label == 'VB':
            node.label = 'VP'


def prune_wordless(ptb_file):
    # Trim sentences with no lexical words
    for sent in ptb_file.children():
        words = [w for w in sent.listWords() if not w.isTrace()]
        if not words:
            ptb_file.detachChild(sent)


def main(out_dir):
    ptb_loc = Path('/usr/local/data/Penn3/parsed/mrg/swbd')
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    for section in ['2', '3', '4']:
        out_section = out_dir.join(section)
        section = ptb_loc.join(section)
        if not out_section.exists():
            out_section.mkdir()
        for mrg_loc in section:
            if not mrg_loc.parts[-1].endswith('.mrg'):
                continue
            dps_loc = str(mrg_loc).replace('parsed/mrg', 'dysfl/dps').replace('mrg', 'dps')
            print mrg_loc
            ptb_file = PTBFile(path=str(mrg_loc))
            dps_strings = [open(dps_loc).read().split('SpeakerA1/SYM', 1)[0].strip()]
            for i, sent in enumerate(ptb_file.children()):
                if i == 0:
                    continue
                if sent.getWord(0).text.startswith('Speaker') and sent.getWord(0).label == 'SYM':
                    dps_strings.append(u'')
                #disfl = mark_disfl(sent)
                dps_string, disfl = get_dps_string(sent)
                dps_strings.append(dps_string)
                clean_sent(sent, disfl)
            prune_wordless(ptb_file)
            out_section.join(mrg_loc.parts[-1].replace('mrg', 'dps')).open('w').write(u'\n'.join(dps_strings)) 
            out_file = out_section.join(mrg_loc.parts[-1]).open('w')
            for sent in ptb_file.children():
                out_file.write(u'( ' + str(sent) + u' )')
                out_file.write(u'\n')
            out_file.close()
    

if __name__ == '__main__':
    plac.call(main)
