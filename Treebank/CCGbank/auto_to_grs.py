"""
Generate Grammatical Relations from a Markedup file and an auto file
"""
import re
from collections import defaultdict
import os

import ccg.category
import Treebank.CCGbank

CONSTRAINT_GROUPS = {
    '=aux': set("""
ai,am,are,be,been,being,is,was,were,'s,'m,'re,has,have,had,'ve,do,did,
does,'d,'ll,ca,can,could,may,might,must,ought,shall,should,will,wo,would,
get,gotten,getting""".replace('\n', '').split(',')),
    '=be': set("ai,am,are,be,been,being,is,was,were,'s,'m,'re".split(',')),
    '=det': set("another,other,some,such".split(',')),
    '=part': set("all,both"),
    None: set()
    }
class Markedup(object):
    """
    Provides a look up table for entries
    """
    comment_strip_re = re.compile(r'#[^\n]+')
    def __init__(self, loc):
        data = open(loc).read()
        header, data = data.split('# now list the markedup categories')
        self.cats = {}
        for entry_str in data.split('\n\n'):
            entry_str = self.comment_strip_re.sub('', entry_str)
            entry_str = entry_str.replace('\n\n', '\n')
            entry_str = entry_str.strip()
            if not entry_str:
                continue
            try:
                entry = MarkedupEntry(entry_str.strip())
            except:
                print repr(entry_str)
                raise
            self.cats[str(entry.category)] = entry



class MarkedupEntry(object):
    """
    A markedup entry that generates Grammatical Relations from dependencies
    """
    def __init__(self, markedup_str):
        self.string = markedup_str
        lines = [l for l in markedup_str.split('\n')
                if not l.strip().startswith('#')]
        bare_category = lines.pop(0)
        n_slots, annotated_category = lines.pop(0).strip().split(' ')
        if lines and lines[0].startswith('  !'):
            alt_markedup = lines.pop(0)[4:]
        else:
            alt_markedup = ''
        slots = defaultdict(list)
        for line in lines:
            slot = Slot(line)
            slots[slot.n].append(slot)

        self.category = ccg.category.from_string(bare_category)
        self.annotated = ccg.category.from_string(annotated_category)
        self.n_grs = int(n_slots)
        if alt_markedup:
            self.alt_annotated = ccg.category.from_string(alt_markedup)
        else:
            self.alt_annotated = self.annotated
        self.grs = slots

    slot_re = re.compile(r'\{([A-Z_])\*?\}<(\d)>')
    def _make_arg_to_var_map(self, annotated):
        var_map = {'%f': '_'}
        for var, slot in self.slot_re.findall(annotated):
            var_map['%' + str(slot)] = var
        return var_map

    def output_grs(self, cat):
        arg_id_to_vars = self._make_arg_to_var_map(
                self.alt_annotated.annotated)
        vars_to_words = cat.map_letters_to_words()
        reduced_map = {}
        for var, words in vars_to_words.items():
            if words:
                reduced_map[var] = words[0]
            else:
                reduced_map[var] = None
        vars_to_words = reduced_map
        head_word = vars_to_words['_']
        gr_strs = []
        for grs_at_slot in self.grs.values():
            gr = self.find_matching_gr(grs_at_slot, arg_id_to_vars, vars_to_words)
            if gr:
                gr_str = gr.output_gr(arg_id_to_vars, vars_to_words)
                if gr_str is not None:
                    gr_strs.append(gr_str)
        return gr_strs

    def find_matching_gr(self, grs, arg_id_to_vars, vars_to_words):
        # Sort/reverse ensures that None constraints get
        # considered last. Note that we break after we match
        # a constraint, so first matching slot is used.
        grs = list(grs)
        grs.sort(key=lambda s: s.constraint_name)
        grs.reverse()
        for gr in grs:
            if gr.passes_constraint(arg_id_to_vars, vars_to_words):
                return gr
        return None

    def output_gr(self, slot_num, group, vars_to_words, use_alt=True):
        arg_id_to_vars = self._make_arg_to_var_map(
                self.alt_annotated.annotated)

        head_word = vars_to_words['_']
        if head_word not in CONSTRAINT_GROUPS[group]:
            group = None
        gr = self.find_matching_gr(self.grs[slot_num], arg_id_to_vars, vars_to_words)
        return gr.output_gr(arg_id_to_vars, vars_to_words)


class Slot(object):
    def __init__(self, slot_str):
        pieces = slot_str.strip().split(' ')
        if pieces and pieces[-1].startswith('='):
            self.constraint_name = pieces.pop(-1)
            self.constraint_group = CONSTRAINT_GROUPS.get(self.constraint_name, set())
        else:
            self.constraint_name = None
            self.constraint_group = set()

        if not pieces[-1].startswith('%') and pieces[-1] != 'ignore':
            self.subtype2 = pieces.pop(-1)
        else:
            self.subtype2 = None

        self.words = [p for p in pieces if p.startswith('%')]
        pieces = [p for p in pieces if not p.startswith('%')]

        self.n = int(pieces.pop(0))
        self.label = pieces.pop(0)
        if pieces:
            self.subtype1 = pieces.pop(0)
        else:
            self.subtype1 = None
        assert not pieces

    def get_arg(self, arg_id_to_vars, vars_to_words):
        return vars_to_words[arg_id_to_vars['%' + str(self.n)]]

    def passes_constraint(self, arg_id_to_vars, vars_to_words):
        if not self.constraint_name:
            return True
        head_word = vars_to_words['_']
        if head_word.text in self.constraint_group:
            return True
        # For supertag constraints e.g. =PP/NP
        arg_word = self.get_arg(arg_id_to_vars, vars_to_words)
        if arg_word and arg_word.stag == self.constraint_name[1:]:
            return True
        return False

    def output_gr(self, arg_id_to_vars, vars_to_words):
        words = []
        arg_word = self.get_arg(arg_id_to_vars, vars_to_words)
        if not arg_word:
            return None
        head_word = vars_to_words['_']
        # The arg id's are things like %f, %l, %2 etc
        for arg_id in self.words:
            if arg_id == '%l':
                word = arg_word
            elif arg_id == '%f':
                word = head_word
            elif arg_id in ['%k', '%c']:
                assert arg_word.stag.argument
                k_heads = arg_word.stag.heads(arg_word.stag.argument)
                word = k_heads[0] if k_heads else None
            else:
                # Vars are _, Y, Z etc
                # Words are persuade_10, Pierre_1 etc
                word = vars_to_words[arg_id_to_vars[arg_id]]

            if isinstance(word, Treebank.CCGbank.CCGLeaf):
                word = '%s_%d' % (word.text, word.wordID)
            elif word is None:
                return None
            words.append(word)
        to_format = [self.label] + words
        if self.subtype1 is not None:
            to_format.insert(1, self.subtype1)
        if self.subtype2 is not None:
            to_format.append(self.subtype2)
        return '(' + ' '.join(to_format) + ')'


def auto_to_grs(markedup, id_, string=None, node=None):
    if string is not None:
        sentence = Treebank.CCGbank.CCGSentence(string=string, globalID=id_, localID=0)
    elif node is not None:
        sentence = node
    else:
        raise Exception
    sentence.unify_vars()
    grs = []
    for word in sentence.listWords():
        mu_entry = markedup.cats.get(str(word.stag))
        if mu_entry is not None:
            grs.extend(mu_entry.output_grs(word.stag))
    return grs

def test_markedup_parsing_no_alt1():
    entry_str = """((S\NP)\(S\NP))/NP
  2 (((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_}/NP{W}<2>){_}
  1 ncmod _ %f %l
  2 dobj %l %f"""
    entry = MarkedupEntry(entry_str)
    assert str(entry.category) == r'((S\NP)\(S\NP))/NP'
    assert entry.annotated.annotated == r'(((S[X]{Y}\NP{Z}){Y}\(S[X]{Y}<1>\NP{Z}){Y}){_}/NP{W}<2>){_}'

def test_markedup_parsing_alt1():
    entry_str = """(S[dcl]\NP)/(S[b]\NP)
  2 ((S[dcl]{_}\NP{Y}<1>){_}/(S[b]{Z}<2>\NP{Y*}){Z}){_}
  ! ((S[dcl]{Z}\NP{Y}<1>){Z}/(S[b]{Z}<2>\NP{Y*}){Z}){_}
  1 ignore =aux
  1 ncsubj %l %f _
  2 aux %f %l =aux
  2 xcomp _ %l %f"""
    entry = MarkedupEntry(entry_str)
    assert entry.category == r'(S[dcl]\NP)/(S[b]\NP)'
    assert entry.alt_annotated.annotated == r'((S[dcl]{Z}\NP{Y}<1>){Z}/(S[b]{Z}<2>\NP{Y*}){Z}){_}'
    assert entry.annotated.annotated == r'((S[dcl]{_}\NP{Y}<1>){_}/(S[b]{Z}<2>\NP{Y*}){Z}){_}'

def test_slot_parsing_subtype_last1():
    slot_str = r'1 ncsubj %l %f _'
    slot = Slot(slot_str)
    assert slot.label == 'ncsubj'
    assert slot.subtype2 == '_', slot.subtype2

def test_slot_parsing_subtype_first1():
    slot_str = r'  2 xcomp _ %l %f =be'
    slot = Slot(slot_str)
    assert slot.label == 'xcomp'
    assert slot.subtype1 == '_'
    assert slot.constraint_name == '=be'

def test_slot_parsing_no_subtype_no_constraint1():
    slot_str = r'  1 det %f %l'
    slot = Slot(slot_str)
    assert slot.label == 'det'
    assert slot.n == 1
    assert slot.constraint_name == None
    assert slot.subtype1 == None

def test_slot_triple1():
    slot_str = r'1 cmod %l %f %2'
    slot = Slot(slot_str)
    assert slot.n == 1
    assert slot.constraint_name == None
    assert slot.words[-1] == '%2'


def test_output1():
    entry_str = """(S[dcl]\NP)/S[dcl]
  2 ((S[dcl]{_}\NP{Y}<1>){_}/S[dcl]{Z}<2>){_}
  1 ncsubj %l %f _
  2 ccomp _ %l %f"""
    entry = MarkedupEntry(entry_str)
    vars_to_words = {
            '_': 'say_2',
            'Y': 'I_1',
            'Z': 'jump_5'
    }
    gr_str = entry.output_gr(2, None, vars_to_words)
    assert gr_str == r'(ccomp _ jump_5 say_2)', gr_str

def test_output2():
    entry_str = """(S[dcl]\NP)/(S[b]\NP)
  2 ((S[dcl]{_}\NP{Y}<1>){_}/(S[b]{Z}<2>\NP{Y*}){Z}){_}
  ! ((S[dcl]{Z}\NP{Y}<1>){Z}/(S[b]{Z}<2>\NP{Y*}){Z}){_}
  1 ignore
  2 aux %f %l"""
    entry = MarkedupEntry(entry_str)
    aux_vars_to_words = {
            '_': 'am',
            'Y': 'I',
            'Z': 'running'
    }
    gr_str = entry.output_gr(2, None, aux_vars_to_words)
    assert '(aux am running)' == gr_str, gr_str


def test_sentence1():
    sent_str = """(<T S[dcl] 0 2> (<T S[dcl] 1 2> (<T NP 0 1> (<T N 1 2> (<L N/N NNP NNP Ms. N_254/N_254>) (<L N NNP NNP Haag N>) ) ) (<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/NP VBZ VBZ plays (S[dcl]\NP_241)/NP_242>) (<T NP 0 1> (<L N NNP NNP Elianti N>) ) ) ) (<L . . . . .>) )"""
    ccg.lexicon.load('/data/ccgbanks/hock07cl/markedup')
    markedup = Markedup('/data/ccgbanks/hock07cl/markedup')
    grs = auto_to_grs(markedup, 'wsj_0200.1', string=sent_str)
    assert grs == ['(ncmod _ Ms._0 Haag_1)',
                   '(ncsubj Haag_1 plays_2 _)',
                   '(dobj Elianti_3 plays_2)']

def test_corpus1():
    loc = '/data/ccgbanks/hock07cl'
    ccgbank = Treebank.CCGbank.CCGbank(loc)
    markedup = Markedup(os.path.join(loc, 'markedup'))
    for file_ in ccgbank.children():
        for sentence in file_.children():
            auto_to_grs(markedup, sentence.globalID, node=sentence)

