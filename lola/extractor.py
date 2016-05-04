"""
:Authors: - Wilker Aziz
"""

from lola.corpus import Corpus
import numpy as np


class FeatureExtractor:

    def extract(self, e_snt, f_snt, i, j):
        """

        :param e_snt:
        :param f_snt:
        :param i:
        :param j:
        :return: list of features
        """
        pass


class WordFeatures(FeatureExtractor):

    def __init__(self, e_corpus, f_corpus):
        self._e_corpus = e_corpus
        self._f_corpus = f_corpus

    def extract(self, e_snt, f_snt, i, j):
        features = list()

        # FRENCH

        # current word
        features.append('f[j]=%s' % self._f_corpus.translate(f_snt[j]))
        # previous word
        if j > 0:
            features.append('f[j-1]=%s' % self._f_corpus.translate(f_snt[j - 1]))
        else:
            features.append('f[j-1]=<BOS>')
        # next word
        if j < len(f_snt) - 1:
            features.append('f[j+1]=%s' % self._f_corpus.translate(f_snt[j + 1]))
        else:
            features.append('f[j+1]=<EOS>')

        # ENGLISH

        # current word
        features.append('e[i]=%s' % self._e_corpus.translate(e_snt[i]))
        # previous word
        if i > 1:  # remember that e_snt[0] is the NULL word
            features.append('e[i-1]=%s' % self._e_corpus.translate(e_snt[i - 1]))
        else:
            features.append('e[i-1]=<BOS>')
        # next word
        if i < len(e_snt) - 1:
            features.append('e[i+1]=%s' % self._e_corpus.translate(e_snt[i + 1]))
        else:
            features.append('e[i+1]=<EOS>')

        # FRENCH-ENGLISH
        features.append('f[j]|e[i]=%s|%s' % (self._f_corpus.translate(f_snt[j]), self._e_corpus.translate(e_snt[i])))

        return features


class AlignmentFeatures(FeatureExtractor):

    def __init__(self):
        pass

    def extract(self, e_snt, f_snt, i, j):
        features = list()

        # Position features
        features.append('j=%d' % (j + 1))  # remember that we need to make the French side 1-based
        features.append('i=%d' % i)  # the English side is 0-based because of NULL
        features.append('floor(j/m)=%d' % np.floor(j/len(f_snt)))
        features.append('floor(i/l)=%d' % np.floor(i/(len(e_snt) - 1)))
        # Jump features
        if i != 0:  # for jumps to words
            features.append('e-jump=%d' % (i - np.floor((j + 1) * (len(e_snt) - 1) / len(f_snt))))
        else:  # for jumps to NULL
            features.append('null-jump=%d' % (i - np.floor((j + 1) * (len(e_snt) - 1) / len(f_snt))))
            features.append('jump-to-null=True')  # this ones fires in general, regardless of the size of the jump
        return features


if __name__ == '__main__':

    F = Corpus('training/example.f')
    E = Corpus('training/example.e', null='<NULL>')

    wfeatures = WordFeatures(E, F)
    afeatures = AlignmentFeatures()

    for s, (f_snt, e_snt) in enumerate(zip(F.itersentences(), E.itersentences()), 1):
        for j in range(len(f_snt)):
            for i in range(len(e_snt)):
                print('# WORD FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j, i))
                for feature in wfeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print('# ALIGNMENT FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j, i))
                for feature in afeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print()
