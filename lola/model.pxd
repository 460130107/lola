cimport numpy as np
from lola.params cimport LexicalParameters


cdef class Model:

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef initialise(self, dict initialiser)


cdef class SufficientStatistics:

    cpdef observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef Model make_model(self)


cdef class IBM1(Model):

    cdef LexicalParameters _lex_parameters

    cpdef LexicalParameters lexical_parameters(self)


cdef class IBM1ExpectedCounts(SufficientStatistics):

    cdef LexicalParameters _lex_counts


"""
cdef class IBM2(Model):

    cdef LexicalParameters _lex_parameters
    # declare a numpy array (memory view)
    cdef np.float_t[::1] _dist_parameters
"""