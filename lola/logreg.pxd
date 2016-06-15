cimport numpy as np
from lola.frepr cimport LexicalFeatureMatrix


cpdef object csr_expected_difference(matrix, row, probs)


cdef class LogisticRegression:

    cdef:
        np.float_t[:,::1] _categoricals

    cpdef float probability(self, int e, int f)

    cpdef np.float_t[::1] categorical(self, int e)