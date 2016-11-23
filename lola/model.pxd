cimport numpy as np
from lola.corpus cimport Corpus
from lola.component cimport GenerativeComponent


cdef class GenerativeModel:

    cdef tuple _components

    cpdef size_t n_components(self)

    cpdef GenerativeComponent component(self, size_t n)

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef observe(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef update(self)

    cpdef load(self, path)

    cpdef save(self, path)
