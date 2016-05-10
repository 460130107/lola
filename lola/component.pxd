cimport numpy as np
from lola.sparse cimport SparseCategorical


cdef class GenerativeComponent:

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef normalise(self)

    cpdef GenerativeComponent zeros(self)


cdef class LexicalParameters(GenerativeComponent):

    cdef list _cpds

    cpdef size_t e_vocab_size(self)

    cpdef size_t f_vocab_size(self)

    cpdef SparseCategorical row(self, int e)


cdef class UniformAlignment(GenerativeComponent):

    pass


cdef class JumpParameters(GenerativeComponent):

    cdef:
        int _max_english_len
        int _max_french_len
        SparseCategorical _categorical

    cpdef int jump(self, int l, int m, int i, int j)

    cpdef int max_english_len(self)

    cpdef int max_french_len(self)


cdef class BrownDistortionParameters(GenerativeComponent):

    cdef:
        int _max_english_len
        float _base_value
        dict _cpds

    cpdef int max_english_len(self)

    cpdef float base_value(self)
