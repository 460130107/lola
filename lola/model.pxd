cimport numpy as np
from lola.component cimport GenerativeComponent


cdef class SufficientStatistics:

    cpdef observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)


cdef class ExpectedCounts(SufficientStatistics):

    cdef list _components

    cpdef list components(self)


cdef class Model:

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef initialise(self, dict initialiser)

    cpdef SufficientStatistics suffstats(self)

    cpdef update(self, list components)


cdef class GenerativeModel(Model):

    cdef list _components

    cpdef size_t n_components(self)

    cpdef GenerativeComponent component(self, int i)


