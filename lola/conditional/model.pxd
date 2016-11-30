cimport numpy as np
from lola.conditional.component cimport GenerativeComponent
from lola.ptypes cimport uint_t
from lola.ptypes cimport real_t


cdef class GenerativeModel:

    cdef tuple _components

    cpdef size_t n_components(self)

    cpdef GenerativeComponent component(self, size_t n)

    cpdef real_t likelihood(self, uint_t[::1] e_snt, uint_t[::1] f_snt, size_t i, size_t j)

    cpdef observe(self, uint_t[::1] e_snt, uint_t[::1] f_snt, size_t i, size_t j, real_t p)

    cpdef update(self)

    cpdef load(self, path)

    cpdef save(self, path)
