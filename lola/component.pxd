cimport numpy as np
from lola.corpus cimport Corpus
from lola.sparse cimport SparseCategorical
from lola.sparse cimport CPDTable


cdef class GenerativeComponent:

    cdef str _name

    cpdef str name(self)

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef normalise(self)

    cpdef GenerativeComponent zeros(self)

    cpdef save(self, Corpus e_corpus, Corpus f_corpus, str path)


cdef class LexicalParameters(GenerativeComponent):

    cdef CPDTable _cpds
    cdef size_t _e_vocab_size
    cdef size_t _f_vocab_size


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

    cdef int jump(self, int l, int m, int i, int j)

    cpdef int max_english_len(self)

    cpdef int max_french_len(self)


cdef class BrownDistortionParameters(DistortionParameters):

    cdef:
        int _max_english_len
        float _base_value
        dict _cpds

    cpdef int max_english_len(self)

    cpdef float base_value(self)
