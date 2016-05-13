cimport numpy as np
from lola.corpus cimport Corpus
from lola.component cimport GenerativeComponent


cdef class SufficientStatistics:

    cpdef void observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p)

    cpdef list components(self)


cdef class DefaultSufficientStatics(SufficientStatistics):

    cdef list _components


cdef class GenerativeModel:

    cpdef size_t n_components(self)

    cpdef GenerativeComponent component(self, size_t n)

    cpdef list components(self)

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef initialise(self, dict initialiser)

    cpdef SufficientStatistics suffstats(self)

    cpdef update(self, list components)


cpdef save_model(GenerativeModel model, Corpus e_corpus, Corpus f_corpus, str path)

cdef class DefaultModel(GenerativeModel):

    cdef list _components


