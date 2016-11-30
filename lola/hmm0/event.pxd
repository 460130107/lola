"""
Event spaces are used in order to make generative components more modular.
An event space is responsible for expressing an event in terms of a (possibly empty) conditioning context and an outcome.

"""

cimport numpy as np
from lola.corpus cimport Corpus
from lola.ptypes cimport uint_t


cdef class EventSpace:

    cdef readonly tuple shape

    cpdef tuple get(self, uint_t[::1] e_snt, uint_t[::1] f_snt, size_t i, size_t j)

    cpdef tuple readable(self, tuple event)


cdef class DummyEventSpace(EventSpace):
    pass


cdef class LexEventSpace(EventSpace):

    cdef:
        Corpus _e_corpus
        Corpus _f_corpus


cdef class JumpEventSpace(EventSpace):

    cdef size_t _longest

