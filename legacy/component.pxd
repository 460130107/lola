cimport numpy as np
from legacy.corpus cimport Corpus
from legacy.sparse cimport SparseCategorical
from legacy.sparse cimport CPDTable
from legacy.event cimport EventSpace
from legacy.event cimport LexEventSpace
from legacy.event cimport JumpEventSpace
from legacy.event cimport DistEventSpace


cdef class GenerativeComponent:

    cdef str _name

    cpdef str name(self)

    cpdef EventSpace event_space(self)

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef normalise(self)

    cpdef GenerativeComponent zeros(self)

    cpdef save(self, Corpus e_corpus, Corpus f_corpus, str path)


cdef class LexicalParameters(GenerativeComponent):

    cdef CPDTable _cpds
    cdef size_t _e_vocab_size
    cdef size_t _f_vocab_size
    cdef LexEventSpace _event_space


    cpdef size_t e_vocab_size(self)

    cpdef size_t f_vocab_size(self)


cdef class DistortionParameters(GenerativeComponent):

    pass

cdef class UniformAlignment(DistortionParameters):

    pass


cdef class JumpParameters(DistortionParameters):

    cdef:
        int _max_english_len
        int _max_french_len
        SparseCategorical _categorical
        JumpEventSpace _event_space

    cdef int jump(self, int l, int m, int i, int j)

    cpdef int max_english_len(self)

    cpdef int max_french_len(self)


cdef class BrownDistortionParameters(DistortionParameters):

    cdef:
        int _max_english_len
        float _base_value
        dict _cpds
        DistEventSpace _event_space

    cpdef int max_english_len(self)

    cpdef float base_value(self)
