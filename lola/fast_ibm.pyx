"""
The default implementation of a generative model can deal with an arbitrary number of generative components.
That is very general and handy, but under-optimised.
In this module, I implement faster variants of IBM models where the only optimisation
is to hard-code generative components.

Here "hard-coding" is not as bad as it sounds, since IBM models are indeed made of a known fixed number
 of generative components.

:Authors: - Wilker Aziz
"""

from lola.corpus cimport Corpus
from lola.model cimport GenerativeModel, SufficientStatistics
from lola.component cimport GenerativeComponent, LexicalParameters, DistortionParameters
from lola.component cimport JumpParameters, BrownDistortionParameters
cimport numpy as np


cdef class IBM1ExpectedCounts(SufficientStatistics):

    cdef LexicalParameters _lex_counts

    def __init__(self, LexicalParameters counts):
        self._lex_counts = counts

    cpdef void observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """
        Account for a potential observation.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :param p: probability of observation, i.e., normalised posterior p(a_j=i | f, e)
        """
        self._lex_counts.plus_equals(e_snt, f_snt, i, j, p)

    cpdef list components(self):
        return [self._lex_counts]


cdef class IBM1(GenerativeModel):
    """
    An IBM1 is a 0th-order model with lexical parameters only.
    """

    cdef LexicalParameters _lex
    cdef LexicalParameters _lex_count

    def __init__(self, LexicalParameters lex_parameters):
        self._lex = lex_parameters

    cpdef size_t n_components(self):
        return 1

    cpdef GenerativeComponent component(self, size_t n):
        return self._lex

    cpdef list components(self):
        return [self._lex]

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: p(a_j = i, f | e)
        """
        return 1.0 / e_snt.shape[0] * self._lex.get(e_snt, f_snt, i, j)

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i | f, e) up to a normalisation constant.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: Z * p(a_j = i | f, e)
        """
        return self._lex.get(e_snt, f_snt, i, j)

    cpdef SufficientStatistics suffstats(self):
        return IBM1ExpectedCounts(self._lex.zeros())

    cpdef update(self, list components):
        self._lex = components[0]

    cpdef LexicalParameters lexical_parameters(self):
        return self._lex


cdef class IBM2ExpectedCounts(SufficientStatistics):

    cdef LexicalParameters _lex_counts
    cdef DistortionParameters _dist_counts

    def __init__(self, LexicalParameters lex_counts, DistortionParameters dist_counts):
        self._lex_counts = lex_counts
        self._dist_counts = dist_counts

    cpdef void observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """
        Account for a potential observation.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :param p: probability of observation, i.e., normalised posterior p(a_j=i | f, e)
        """
        self._lex_counts.plus_equals(e_snt, f_snt, i, j, p)
        self._dist_counts.plus_equals(e_snt, f_snt, i, j, p)

    cpdef list components(self):
        return [self._lex_counts, self._dist_counts]


cdef class IBM2(GenerativeModel):
    """
    An IBM1 is a 0th-order model with lexical parameters only.
    """

    cdef LexicalParameters _lex
    cdef DistortionParameters _dist

    def __init__(self, LexicalParameters lex_parameters, DistortionParameters dist_parameters):
        self._lex = lex_parameters
        self._dist = dist_parameters

    cpdef size_t n_components(self):
        return 2

    cpdef GenerativeComponent component(self, size_t n):
        if n == 0:
            return self._lex
        elif n == 1:
            return self._dist
        else:
            raise ValueError('I only have 2 components')

    cpdef list components(self):
        return [self._lex, self._dist]

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: p(a_j = i, f | e)
        """
        return self._lex.get(e_snt, f_snt, i, j) * self._dist.get(e_snt, f_snt, i, j)

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i | f, e) up to a normalisation constant.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: Z * p(a_j = i | f, e)
        """
        return self._lex.get(e_snt, f_snt, i, j) * self._dist.get(e_snt, f_snt, i, j)

    cpdef SufficientStatistics suffstats(self):
        return IBM2ExpectedCounts(self._lex.zeros(), self._dist.zeros())

    cpdef update(self, list components):
        self._lex = components[0]
        self._dist = components[1]

    cpdef initialise(self, dict initialiser):
        if 'IBM1' in initialiser:
            # we are replacing our own lexical parameters, by those of an IBM1 which has already been optimised
            self._lex = initialiser['IBM1'].lexical_parameters()

    cpdef LexicalParameters lexical_parameters(self):
        return self._lex

    cpdef DistortionParameters distortion_parameters(self):
        return self._dist


cdef class VogelIBM2(IBM2):
    """
    An IBM1 is a 0th-order model with lexical parameters only.
    """

    def __init__(self, LexicalParameters lex_parameters, JumpParameters dist_parameters):
        super(VogelIBM2, self).__init__(lex_parameters, dist_parameters)


cdef class BrownIBM2(IBM2):
    """
    An IBM1 is a 0th-order model with lexical parameters only.
    """

    def __init__(self, LexicalParameters lex_parameters, BrownDistortionParameters dist_parameters):
        super(BrownIBM2, self).__init__(lex_parameters, dist_parameters)

