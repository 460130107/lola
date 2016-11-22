"""
Event spaces are used in order to make generative components more modular.
An event space is responsible for expressing an event in terms of a (possibly empty) conditioning context and an outcome.

"""

cimport numpy as np
from lola.corpus cimport Corpus


cdef class EventSpace:

    cdef readonly tuple shape

    cpdef tuple get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef tuple readable(self, tuple event)


cdef class DummyEventSpace(EventSpace):
    pass


cdef class LexEventSpace(EventSpace):

    cdef:
        Corpus _e_corpus
        Corpus _f_corpus


cdef class JumpEventSpace(EventSpace):

    cdef size_t _shift

