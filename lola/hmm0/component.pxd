from lola.sparse cimport CPDTable
from lola.hmm0.event cimport EventSpace
from lola.ptypes cimport uint_t
from lola.ptypes cimport real_t


cpdef float cmp_prob(tuple pair)


cdef class GenerativeComponent:

    cdef readonly name

    cdef readonly EventSpace event_space

    cpdef real_t prob(self, uint_t[::1] e_snt, uint_t[::1] f_snt, size_t i, size_t j)

    cpdef observe(self, uint_t[::1] e_snt, uint_t[::1] f_snt, size_t i, size_t j, real_t p)

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
