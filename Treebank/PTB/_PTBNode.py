import re
import bisect

from Treebank.Nodes import Node


class PTBNode(Node):
    """
    A node in a parse tree
    """
    _labelRE = re.compile(r'([^-=]+)(?:-([^-=\d]+))?(-UNF)?(?:-(\d+))?(?:=(\d+))?')
    def __init__(self, **kwargs):
        string = kwargs.pop('string', None)
        if string is not None:        
            pieces = PTBNode._labelRE.match(string).groups()
            label, functionLabel, unf, identifier, identified = pieces
        else:
            label = kwargs.pop('label')
            functionLabel = kwargs.pop('functionLabel', None)
            identifier = kwargs.pop('identifier', None)
            identified = kwargs.pop('identified', None)
            unf = kwargs.pop('unf', False)
        if kwargs:
            raise StandardError, kwargs
        if functionLabel == 'UNF':
            functionLabel = None
            unf = True
        self.functionLabel = functionLabel
        self.identifier = identifier
        self.identified = identified
        self.unf = bool(unf)
        # This will be filled by a node that a trace indicates
        self.traced = None
        Node.__init__(self, label)

    def head(self):
        """
        This should really implement the Magerman head-finding
        heuristics, but currently selects last word
        """
        # TODO Implement Magerman head finding
        return self.getWord(-1)

