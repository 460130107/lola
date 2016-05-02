"""
:Authors: - Wilker Aziz
"""

from lola.model import Model, SufficientStatistics
from lola.params import LexicalParameters
import numpy as np


class IBM2(Model):

    def __init__(self, lex_parameters, dist_parameters):
        self._lex_parameters = lex_parameters
        self._dist_parameters = dist_parameters

    def likelihood(self, e_snt, f_snt,  i,  j):
        """
        p(a_j, f_1^m|e_0^l)

        :param e_snt:
        :param f_snt:
        :param i:
        :param j:
        :return:
        """
        length = 1.0/len(f_snt)
        lexical = self._lex_parameters.get(e_snt[i], f_snt[j])
        # jump = i - np.floor(j * len(f_snt) / len(e_snt))
        # distortion = self._dist_parameters.get(jump)
        # delegate the computation of Jump to the object responsible for distortion parameters
        distortion = self._dist_parameters.get(len(e_snt), len(f_snt), i, j)
        return length * lexical * distortion

    def posterior(self, e_snt, f_snt,  i, j):
        """
        p(a_j | f_1^m, e_0^l) up to a normalisation constant

        :param e_snt:
        :param f_snt:
        :param i:
        :param j:
        :return:
        """
        lexical = self._lex_parameters.get(e_snt[i], f_snt[j])
        distortion = self._dist_parameters.get(len(e_snt), len(f_snt), i, j)
        return lexical * distortion

    def initialise(self, initialiser):
        """

        :param initialiser: a dictionary containing other models (already trained ones)
        :return:
        """
        if 'IBM1' in initialiser:
            # we are replacing our own lexical parameters, by those of an IBM1 which has already been optimised
            self._lex_parameters = initialiser['IBM1'].lexical_parameters()


class IBM2ExpectedCounts(SufficientStatistics):

    def __init__(self, e_vocab_size, f_vocab_size, max_english_len, max_french_len):
        self._lex_counts = LexicalParameters(e_vocab_size, f_vocab_size, 0.0)
        self._dist_counts = JumpParameters(max_english_len, max_french_len, 0.0)

    def observation(self, e_snt, f_snt, i, j, p):
        # we count a Lexical event
        # that is, French word at position j being generated by English word at position i
        # with probability p
        self._lex_counts.plus_equals(e_snt[i], f_snt[j], p)
        # we also count a Distortion event
        # that is, a certain type of jump has occurred
        self._dist_counts.plus_equals(len(e_snt), len(f_snt), i, j, p)

    def make_model(self):
        """
        Maximise our independent models and reset the counts.
        :return: new IBM2
        """
        self._lex_counts.normalise()
        self._dist_counts.normalise()
        model = IBM2(self._lex_counts, self._dist_counts)
        # TODO: reset counts
        self._lex_counts = LexicalParameters(self._lex_counts.e_vocab_size(),
                                             self._lex_counts.f_vocab_size(),
                                             0.0)
        self._dist_counts = JumpParameters(self._dist_counts.max_english_len(),
                                           self._dist_counts.max_french_len(),
                                           0.0)
        return model


class JumpParameters:

    def __init__(self, max_english_len, max_french_len, base_value):
        self._max_english_len = max_english_len
        self._max_french_len = max_french_len
        self._base_value = base_value
        self._categorical = dict()

    def jump(self, l, m, i, j):
        return i - np.floor((j + 1) * m / l)

    def get(self, l, m, i, j):
        # compute the jump
        jump = self.jump(l, m, i, j)
        # retrieve the parameter associated with it
        # or a base value in case the jump is not yet mapped
        return self._categorical.get(jump, self._base_value)

    def max_english_len(self):
        return self._max_english_len

    def max_french_len(self):
        return self._max_french_len

    def normalise(self):
        # sum the values already mapped
        Z = sum(self._categorical.values())
        # sum the values not yet mapped (those with a base value)
        total_size = self._max_english_len + self._max_french_len + 1
        unmapped = total_size - len(self._categorical)
        Z += unmapped * self._base_value
        # compute a new categorical with normalised values
        categorical = dict()
        for jump, value in self._categorical.items():
            categorical[jump] = value / Z
        # and keep this new one
        self._categorical = categorical

    def plus_equals(self, l, m, i, j, p):
        # compute the jump
        value = self.get(l, m, i, j)
        jump = self.jump(l, m, i, j)
        updated = value + p
        self._categorical[jump] = updated
        return updated
