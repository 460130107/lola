from lola.params import LexicalParameters
from lola.model import Model, SufficientStatistics
import numpy as np

class J_IBM2(Model):

    def __init__(self, dist_parameters, lex_parameters):
        """

        :param dist_parameters:
        :param lex_parameters:
        """
        self._dist_parameters = dist_parameters
        self._lex_parameters = lex_parameters

    def likelihood(self, e_snt, f_snt, i, j):
        """

        :param e_snt:
        :param f_snt:
        :param i:
        :param j:
        :return:
        """
        m = 1.0/len(f_snt)
        lex_decision = self._lex_parameters.get(e_snt[i], f_snt[j])
        dist_decision = self._dist_parameters.get(e_snt, f_snt, i, j)
        return m * lex_decision * dist_decision

    def posterior(self, e_snt, f_snt, i, j):
        """

        :param e_snt:
        :param f_snt:
        :param i:
        :param j:
        :return:
        """
        lex_decision = self._lex_parameters.get(e_snt[i], f_snt[j])
        dist_decision = self._dist_parameters.get(e_snt, f_snt, i, j)
        return lex_decision * dist_decision

class ExpectedCountsIBM2(SufficientStatistics):

    def __init__(self, e_max_len, f_max_len, e_vocab_size, f_vocab_size):
        self._dist_counts = VogelDistortionParameters(e_max_len, f_max_len, 0.0)
        self._lex_counts = LexicalParameters(e_vocab_size, f_vocab_size, 0.0)

    def observation(self, e_snt, f_snt, i, j, value):
        self._dist_counts.plus_equals(len(e_snt), len(f_snt), i, j, value)
        self._lex_counts.plus_equals(e_snt[i], f_snt[i], value)

    def make_model(self):
        self._dist_counts.normalise()
        self._lex_counts.normalise()
        self._dist_counts = VogelDistortionParameters(self._dist_counts.e_max_len(),
                                                     self._dist_counts.f_max_len(),
                                                     0.0)
        self._lex_counts = LexicalParameters(self._lex_counts.e_vocab_size(),
                                             self._lex_counts.f_vocab_size(),
                                             0.0)


class VogelDistortionParameters:

    def __init__(self, e_max_len, f_max_len, p):
        length = e_max_len + f_max_len + 1
        self._categorical = np.full((length, 1), p)
        self._e_max_len = e_max_len
        self._f_max_len = f_max_len

    def e_max_len(self):
        return self._e_max_len

    def f_max_len(self):
        return self._f_max_len

    def jump(self, l, m, i, j):
        return i - np.floor(j * l / m)

    def index_jump(self, l, m, i, j):
        jump = self.jump(l, m, i, j)
        return jump + self._f_max_len

    def get(self, l, m, i, j):
        index_jump = self.index_jump(l, m, i, j)
        return self._categorical[index_jump]

    def normalise(self):
        z = self._categorical.sum()
        self._categorical /= z

    def plus_equals(self, l, m, i, j, p):
        index_jump = self.index_jump(l, m, i, j)
        updated = self._categorical[index_jump] + p
        self._categorical[index_jump] = updated
        return updated