"""
:Authors: - Wilker Aziz
"""

from lola.corpus import Corpus
import numpy as np
import re


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

        # ALIGNMENT DIAGONAL CLOSENESS (based on Blunsom and Cohn (2006)) and described in Dyer et al. (2013)
        if i != 0:
            features.append('|(i/m)-(j/l)|=%f' % (abs((i / len(f_snt)) - j / (len(e_snt) - 1)))) # Continuous feature
        else:
            
            features.append('|(i/m)-(j/l)|=%d' % 0) # TODO: instead of 0 something else
        return features


class DistanceFeatures(FeatureExtractor):

    def __init_(self):
        pass

    def extract(self, e_snt, f_snt, i, j):
        features = list()

        distance = (j + 1) - i # make French side 1-based
        if i != 0: # Only if jump not to NULL
            features.append('j-i=%d' % distance)
        else:
            features.append('j-i=%d' % 0) # TODO: What is good value if i=0

        if i == 0:
            jumptype = 'NULL'
        elif distance > 0:
            jumptype = 'FORWARD'
        elif distance < 0:
            jumptype = 'BACKWARD'
        else:
            jumptype = 'STAY'

        features.append('jumptype=%s' % jumptype)

        return features


class SentenceLengthFeatures(FeatureExtractor):
    # Motivation: sentence length can have influence on the lexical entries (e.g. words or morphology)
    # in a sentence. (e.g. shorter sentence length can be compensated by richer morphology)

    def __init__(self, e_corpus, f_corpus):
        self._e_corpus = e_corpus
        self._f_corpus = f_corpus

    def extract(self, e_snt, f_snt, i, j):
        features = list()

        l = len(e_snt) - 1
        m = len(f_snt)

        # SENTENCE LENGTH CONJOINED WITH LEXICAL ENTRY
        features.append('m|f[j]=%d|%s' % (m, self._f_corpus.translate(f_snt[j])))
        features.append('l|e[i]=%d|%s' % (l, self._e_corpus.translate(e_snt[i])))

        return features


class BigramFeatures(FeatureExtractor):
    def __init__(self, e_corpus, f_corpus):
        self._e_corpus = e_corpus
        self._f_corpus = f_corpus

    def extract(self, e_snt, f_snt, i, j):
        features = list()

        # BIGRAM FRENCH
        if j > 0:
            features.append('f[j-1]_f[j]=%s_%s' % (self._f_corpus.translate(f_snt[j - 1]),
                                                   self._f_corpus.translate(f_snt[j])))
        else:
            features.append('f[j-1]_f[j]=<BOS>_%s' % self._f_corpus.translate(f_snt[j]))

        if j < len(f_snt) - 1:
            features.append('f[j]_f[j+1]=%s_%s' % (self._f_corpus.translate(f_snt[j]),
                                                   self._f_corpus.translate(f_snt[j + 1])))
        else:
            features.append('f[j]_f[j+1]=%s_<EOS>' % self._f_corpus.translate(f_snt[j]))

        # BIGRAM ENGLISH
        # TODO: Make exception when i = 0
        if i > 1:
            features.append('e[i-1]_e[i]=%s_%s' % (self._e_corpus.translate(e_snt[i - 1]),
                                                   self._e_corpus.translate(e_snt[i])))
        else:
            features.append('e[i-1]_e[i]=<BOS>_%s' % self._e_corpus.translate(e_snt[i]))

        if i < len(e_snt) - 1:
            features.append('e[i]_e[i+1]=%s_%s' % (self._e_corpus.translate(e_snt[i]),
                                                   self._e_corpus.translate(e_snt[i + 1])))
        else:
            features.append('e[i]_e[i+1]=%s_<EOS>' % self._e_corpus.translate(e_snt[i]))

        return features


class WordOperationFeatures(FeatureExtractor):

    def __init__(self, e_corpus, f_corpus):
        self._e_corpus = e_corpus
        self._f_corpus = f_corpus

    def extract(self, e_snt, f_snt, i, j):
        features = list()

        # FRENCH SUFFIX
        features.append('f[j][-2:]=%s' % self._f_corpus.translate(f_snt[j])[-2:])
        features.append('f[j][-3:]=%s' % self._f_corpus.translate(f_snt[j])[-3:])
        features.append('f[j][-4:]=%s' % self._f_corpus.translate(f_snt[j])[-4:])

        # FRENCH PREFIX
        features.append('f[j][:2]=%s' % self._f_corpus.translate(f_snt[j])[:2])
        features.append('f[j][:3]=%s' % self._f_corpus.translate(f_snt[j])[:3])
        features.append('f[j][:4]=%s' % self._f_corpus.translate(f_snt[j])[:4])

        # ENGLISH SUFFIX
        features.append('e[i][-2:]=%s' % self._e_corpus.translate(e_snt[i])[-2:])
        features.append('e[i][-3:]=%s' % self._e_corpus.translate(e_snt[i])[-3:])
        features.append('e[i][-4:]=%s' % self._e_corpus.translate(e_snt[i])[-4:])

        # ENGLISH PREFIX
        features.append('e[i][:2]=%s' % self._e_corpus.translate(e_snt[i])[:2])
        features.append('e[i][:3]=%s' % self._e_corpus.translate(e_snt[i])[:3])
        features.append('e[i][:4]=%s' % self._e_corpus.translate(e_snt[i])[:4])

        # FRENCH WORD-SUFFIX
        features.append('f[j][:-2]=%s' % self._f_corpus.translate(f_snt[j])[:-2])
        features.append('f[j][:-3]=%s' % self._f_corpus.translate(f_snt[j])[:-3])
        features.append('f[j][:-4]=%s' % self._f_corpus.translate(f_snt[j])[:-4])

        # FRENCH WORD-PREFIX
        features.append('f[j][2:]=%s' % self._f_corpus.translate(f_snt[j])[2:])
        features.append('f[j][3:]=%s' % self._f_corpus.translate(f_snt[j])[3:])
        features.append('f[j][4:]=%s' % self._f_corpus.translate(f_snt[j])[4:])

        # ENGLISH WORD-SUFFIX
        features.append('e[i][:-2]=%s' % self._e_corpus.translate(e_snt[i])[:-2])
        features.append('e[i][:-3]=%s' % self._e_corpus.translate(e_snt[i])[:-3])
        features.append('e[i][:-4]=%s' % self._e_corpus.translate(e_snt[i])[:-4])

        # ENGLISH WORD-PREFIX
        features.append('e[i][2:]=%s' % self._e_corpus.translate(e_snt[i])[2:])
        features.append('e[i][3:]=%s' % self._e_corpus.translate(e_snt[i])[3:])
        features.append('e[i][4:]=%s' % self._e_corpus.translate(e_snt[i])[4:])

        # OTHER STRING OPERATIONS
        digits = re.compile('\d') # Check if french word contains one or more digits
        features.append('French-contains-digits=%s' % bool(digits.search(self._f_corpus.translate(f_snt[j]))))
        features.append('English-contains-digits=%s' % bool(digits.search(self._e_corpus.translate(e_snt[i]))))
        return features


if __name__ == '__main__':

    F = Corpus('training/example.f')
    E = Corpus('training/example.e', null='<NULL>')

    wfeatures = WordFeatures(E, F)
    afeatures = AlignmentFeatures()
    dfeatures = DistanceFeatures()
    sfeatures = SentenceLengthFeatures(E, F)
    bfeatures = BigramFeatures(E, F)
    wofeatures = WordOperationFeatures(E, F)

    for s, (f_snt, e_snt) in enumerate(zip(F.itersentences(), E.itersentences()), 1):
        for j in range(len(f_snt)):
            for i in range(len(e_snt)):
                print('# WORD FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j + 1, i))
                for feature in wfeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print('# ALIGNMENT FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j + 1, i))
                for feature in afeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print('# DISTANCE FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j + 1, i))
                for feature in dfeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print('# SENTENCE LENGTH FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j + 1, i))
                for feature in sfeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print('# BIGRAM FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j + 1, i))
                for feature in bfeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print('# WORD OPERATIONS FEATURES FOR SENTENCE %d AND ALIGNMENT a[%d]=%d' % (s, j + 1, i))
                for feature in wofeatures.extract(e_snt, f_snt, i, j):
                    print(feature)
                print()
