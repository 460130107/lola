"""
:Authors: - Wilker Aziz
"""

import unittest

from lola.corpus import Corpus


class CorpusTestCase(unittest.TestCase):

    def setUp(self):
        self.F = Corpus('../training/example.f')

        self.f_string_corpus = []
        with open('../training/example.f', 'r') as fi:
            for snt in fi:
                self.f_string_corpus.append(snt.split())

        self.E = Corpus('../training/example.e', null='<NULL>')

        self.e_string_corpus = []
        with open('../training/example.e', 'r') as fi:
            for snt in fi:
                self.e_string_corpus.append(['<NULL>'] + snt.split())  # here we decorate with NULL words

    def test_iter(self):
        f_corpus_size = 0
        for f_snt, e_snt in zip(self.F.itersentences(), self.E.itersentences()):
            for j, f in enumerate(f_snt):  # this is a position j and the f word associated with it
                f_corpus_size += 1
                for i, e in enumerate(e_snt):  # this is a position i and the e-word associated with it (it includes the NULL already)
                    pass
        self.assertEqual(f_corpus_size, self.F.corpus_size())

    def test_f(self):
        reconstructed = []
        for f_snt in self.F.itersentences():
            snt_reconstruction = []
            for f_word in f_snt:
                snt_reconstruction.append(self.F.translate(f_word))  # get the string
            reconstructed.append(snt_reconstruction)
        self.assertEqual(reconstructed, self.f_string_corpus, 'F-reconstruction failed')

    def test_f2(self):
        reconstructed = []
        for s in range(self.F.n_sentences()):
            f_snt = self.F.sentence(s)
            snt_reconstruction = []
            for f_word in f_snt:
                snt_reconstruction.append(self.F.translate(f_word))  # get the string
            reconstructed.append(snt_reconstruction)
        self.assertEqual(reconstructed, self.f_string_corpus, 'F-reconstruction failed')

    def test_e(self):
        reconstructed = []
        for e_snt in self.E.itersentences():
            snt_reconstruction = []
            for e_word in e_snt:
                snt_reconstruction.append(self.E.translate(e_word))  # get the string
            reconstructed.append(snt_reconstruction)
        self.assertEqual(reconstructed, self.e_string_corpus, 'E-reconstruction failed')

    def test_e2(self):
        reconstructed = []
        for s in range(self.E.n_sentences()):
            e_snt = self.E.sentence(s)
            snt_reconstruction = []
            for e_word in e_snt:
                snt_reconstruction.append(self.E.translate(e_word))  # get the string
            reconstructed.append(snt_reconstruction)
        self.assertEqual(reconstructed, self.e_string_corpus, 'E-reconstruction failed')


if __name__ == '__main__':
    unittest.main()