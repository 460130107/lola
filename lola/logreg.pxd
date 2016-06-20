cimport numpy as np


cdef class LogisticRegression:

    cdef:
        np.float_t[:,::1] _categoricals

    cpdef float probability(self, int e, int f)

    cpdef np.float_t[::1] categorical(self, int e)

