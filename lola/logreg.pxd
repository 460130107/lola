cimport numpy as np


cdef class LogisticRegression:

    cdef:
        object _feature_matrix
        np.float_t[::1] _weight_vector
        np.float_t[:,::1] _numerator_cache
        np.float_t[::1] _denominator_cache
        size_t _e_vocab_size
        size_t _f_vocab_size

    cpdef float probability(self, int e, int f)

    cdef float _potential(self, int e, int f)

    cdef float potential(self, int e, int f)

    cdef float denominator(self, int e)