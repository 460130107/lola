cimport numpy as np
from lola.sparse cimport CPDTable
from lola.event cimport EventSpace


cdef class GenerativeComponent:

    cdef readonly name

    cdef readonly EventSpace event_space

    cpdef float prob(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float observe(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef setup(self)

    cpdef update(self)

    cpdef load(self, path)

    cpdef save(self, path)


cdef class UniformAlignment(GenerativeComponent):

    pass


cdef class CategoricalComponent(GenerativeComponent):

    cdef:
        CPDTable _cpds
        CPDTable _counts


cdef class BrownLexical(CategoricalComponent):

    pass


cdef class VogelJump(CategoricalComponent):

    pass
