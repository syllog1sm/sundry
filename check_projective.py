import sys

class Token:
    def __init__(self, line):
        fields = line.split()
        self.idx = int(fields[0]) - 1
        self.word = fields[1]
        self.pos = fields[3]
        self.headIdx = int(fields[6]) - 1
        if self.headIdx == -1:
            self.headIdx = self.idx
        self.label = fields[7]

    def __str__(self):
        headIdx = (self.headIdx + 1) if self.headIdx != self.idx else 0
        fields = [str(self.idx + 1), self.word, '-', self.pos, '-', '-', str(headIdx),
                  self.label, '-', '-']
        return '\t'.join(fields)


def is_projective(tokens):
    if len(tokens) < 3:
        return True
    last_children = {}
    arcs = []
    for token in tokens:
        first, second = sorted((token.idx, token.headIdx))
        arcs.append((first, second))
    for arc1 in arcs:
        for arc2 in arcs:
            # Arc1 starts before Arc2, but finishes in between arc2's start and end
            if arc1[0] < arc2[0] < arc1[1] < arc2[1]:
                return False
    return True

def main():
    sent = []
    counter = 0
    for line in sys.stdin:
        if not line.strip():
            if is_projective(sent):
                print '\n'.join(str(token) for token in sent)
                print
            else:
                counter += 1
            sent = []
        else:
            sent.append(Token(line))
    if sent and is_projective(sent):
        print '\n'.join(str(token) for token in sent)
        print
    elif sent:
        counter += 1
    print >> sys.stderr, counter

if __name__ == '__main__':
    main()
