from lola.params import LexicalParameters
from lola.model import SufficientStatistics
import numpy as np


class ExpectedCountsOriginalIBM2(SufficientStatistics):

    def __init__(self, e_max_len, f_max_len, e_vocab_size, f_vocab_size):
        self._dist_counts = DistortionParameters(e_max_len, f_max_len)
        self._dist_counts.initialise_dict(e_max_len, f_max_len, 0.0)
        self._lex_counts = LexicalParameters(e_vocab_size, f_vocab_size, 0.0)


    def observation(self, e_snt, f_snt, i, j, value):
        self._dist_counts.plus_equals(e_snt, f_snt, j, value)
        self._lex_counts.plus_equals(e_snt[i], f_snt[i], value)


    def make_model(self):
        self._dist_counts.normalise()
        self._lex_counts.normalise()
        self._dist_counts.initialise_dict(self._dist_counts.e_max_len(),
                                          self._dist_counts.f_max_len(),
                                          0.0)

        self._lex_counts = LexicalParameters(self._lex_counts.e_vocab_size(),
                                             self._lex_counts.f_vocab_size(),
                                             0.0)


class DistortionParameters:

    def __init__(self, l, m):
        self._distortion_parameters = dict()
        self._e_max_len = l
        self._f_max_len = m

    def initialise_dict(self, l, m, p):
        self._distortion_parameters = dict()
        for e_dict in l:
            self._distortion_parameters[e_dict] = dict()
            for f_dict in m:
                self._distortion_parameters[e_dict][f_dict] = dict()
                for j in f_dict:
                    self._distortion_parameters[e_dict][f_dict][j] = p

    def initialise_uniform(self, l, m):
        self._distortion_parameters = dict()
        for e_dict in l:
            self._distortion_parameters[e_dict] = dict()
            for f_dict in m:
                self._distortion_parameters[e_dict][f_dict] = dict()
                for j in f_dict:
                    self._distortion_parameters[e_dict][f_dict][j] = 1.0/f_dict

    def e_max_len(self):
        return self._e_max_len

    def f_max_len(self):
        return self._f_max_len

    def get(self, l, m, j):
        return self._distortion_parameters[l][m][j]

    def normalise(self):
        for e_snt_len in self._distortion_parameters:
            for f_snt_len_dict in e_snt_len:
                # keep track of current length of french sentence
                current_f_length = 0
                for f_snt_len in f_snt_len_dict:
                    z = f_snt_len.sum()
                    for j, value in f_snt_len.items():
                        self._distortion_parameters[j] = value / z
                current_f_length += 1

    def plus_equals(self, l, m, j, p):
        value = self.get(l, m, j)
        updated = value + p
        self._distortion_parameters[l][m][j] = updated
        return updated
