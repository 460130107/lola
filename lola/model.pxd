"""
Authors: - Wilker Aziz
"""

cimport numpy as np
from lola.sparse cimport LexicalParameters


cdef class Model:

    cpdef float pij(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float count(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef normalise(self)


cdef class IBM1(Model):

    cdef LexicalParameters _lex_parameters
    cdef LexicalParameters _lex_counts


