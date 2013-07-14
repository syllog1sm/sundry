import sys
import re
import os.path

import ccg.lexicon
from Treebank.PTB import PennTreebank
from _CCGNode import CCGNode
from _CCGFile import CCGFile

class CCGbank(PennTreebank, CCGNode):
    fileClass = CCGFile
    def __init__(self, path=None, **kwargs):
        PennTreebank.__init__(self, path, **kwargs)
        ccg.lexicon.load(os.path.join(path, 'markedup'))

    def child(self, index):
        """
        Read a file by zero-index offset
        """
        path = self._children[index]
        print >> sys.stderr, path
        return self.fileClass(path=path)

    def sentence(self, key):
        fileName, sentID = key.split('.')
        section = fileName[4:6]
        fileID = os.path.join(self.path, 'data', 'AUTO', section, fileName +
                              '.auto')
        f = self.file(fileID)
        #pargLoc = fileID.rsplit('/', 2)[0].replace('AUTO', 'PARG')
        #f.addPargDeps(pargLoc)
        return f.sentence(key)

    def tokens(self):
        """
        Generate tokens without parsing the files properly
        """
        tokenRE = re.compile(r'<L (\S+) \S+ (\S+) (\S+) \S+>')
        for path in self._children:
            string = open(path).read()
            for cat, pos, form in tokenRE.findall(string):
                yield form, pos, cat
            
