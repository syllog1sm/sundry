import re

from Treebank.Nodes import Sentence
from _PTBNode import PTBNode
from _PTBLeaf import PTBLeaf

class PTBSentence(PTBNode, Sentence):
    """
    The root of the parse tree
    
    Has no parent, and one or more children
    """
    def __init__(self, **kwargs):
        if 'string' in kwargs:
            node = self._parseString(kwargs.pop('string'))
        elif 'node' in kwargs:
            node = kwargs.pop('node')
        globalID = kwargs.pop('globalID')
        localID = kwargs.pop('localID')
        PTBNode.__init__(self, label='S', **kwargs)
        self.globalID = globalID
        self.localID = localID
        self.attachChild(node)


    bracketsRE = re.compile(r'(\()([^\s\)\(]+)|([^\s\)\(]+)?(\))')
    def _parseString(self, sent_text):
        openBrackets = []
        parentage = {}
        nodes = {}
        nWords = 0
        
        # Get the nodes and record their parents
        for match in self.bracketsRE.finditer(sent_text):
            open_, label, text, close = match.groups()
            if open_:
                assert not close
                assert label
                openBrackets.append((label, match.start()))
            else:
                assert close
                label, start = openBrackets.pop()
                if text:
                    newNode = PTBLeaf(label=label, text=text, wordID=nWords)
                    nWords += 1
                else:
                    newNode = PTBNode(string=label)
                if openBrackets:
                    # Store parent start position
                    parentStart = openBrackets[-1][1]
                    parentage[newNode] = parentStart
                else:
                    top = newNode
                # Organise nodes by start
                nodes[start] = newNode
        try:
            self._connectNodes(nodes, parentage)
        except:
            print sent_text
            raise
        by_identifier = dict((n.identifier, n) for n in top.depthList() if n.identifier)
        for node in top.depthList():
            if node.identified:
                node.traced = by_identifier[node.identified]
        return top
        



    
