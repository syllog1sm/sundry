from Treebank.Nodes import File
from _PTBNode import PTBNode
from _PTBSentence import PTBSentence


class PTBFile(File, PTBNode):
    """
    A Penn Treebank file
    """
    def __init__(self, **kwargs):
        if 'string' in kwargs:
            text = kwargs.pop('string')
            path = kwargs.pop('path')
        else:
            path = kwargs.pop('path')
            text = open(path).read()
        # Sometimes sentences start (( instead of ( (. This is an error, correct it
        filename = path.split('/')[-1]
        self.path = path
        self.filename = filename
        self.ID = filename
        self._IDDict = {}
        PTBNode.__init__(self, label='File', **kwargs)
        self._parseFile(text)
        
            

    def _parseFile(self, text):
        currentSentence = []
        for line in text.strip().split('\n'):
            # Detect the start of sentences by line starting with (
            # This is messy, but it keeps bracket parsing at the sentence level
            if line.startswith('(') and currentSentence:
                self._addSentence(currentSentence)
                currentSentence = []
            currentSentence.append(line)
        self._addSentence(currentSentence)

    def _addSentence(self, lines):
        sentStr = '\n'.join(lines)[1:-1]
        nSents = len(self)+1
        sentID = '%s~%s' % (self.filename, str(nSents).zfill(4))
        self.attachChild(PTBSentence(string=sentStr, globalID=sentID, localID=self.length()))



        
