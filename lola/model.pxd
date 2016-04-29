"""
Authors: - Wilker Aziz
"""

cimport numpy as np
from lola.sparse cimport LexicalParameters


cdef class Model:

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)


cdef class SufficientStatistics:

    cpdef observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef Model make_model(self)


cdef class IBM1(Model):

    cdef LexicalParameters _lex_parameters


cdef class IBM1ExpectedCounts(SufficientStatistics):

    cdef LexicalParameters _lex_counts

