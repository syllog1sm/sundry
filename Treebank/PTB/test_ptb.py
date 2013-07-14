import unittest

import Treebank.PTB

class TestPTB(unittest.TestCase):
    def test_corpus(self):
        path = '/home/matt/code/repos/data/TreeBank3/parsed/mrg/wsj/'
        ptb = Treebank.PTB.PennTreebank(path=path)
        self.assertEqual(ptb.length(), 2312)
        asbestos = ptb.child(2).child(0)
        self.assertEqual(41, len(asbestos.listWords()))

if __name__ == '__main__':
    unittest.main()
