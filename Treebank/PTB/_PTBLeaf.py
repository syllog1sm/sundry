from Treebank.Nodes import Leaf
from _PTBNode import PTBNode

class PTBLeaf(Leaf, PTBNode):
    def __init__(self, **kwargs):
        self.wordID = kwargs.pop('wordID')
        self.text = kwargs.pop('text')
        if self.text.startswith('*ICH*'):
            kwargs['identified'] = self.text.split('-')[1]
        self.synsets = []
        self.supersenses = []
        self.lemma = self.text
        PTBNode.__init__(self, **kwargs)


