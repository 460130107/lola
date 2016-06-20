from lola.corpus cimport Corpus
from lola.component cimport GenerativeComponent
from lola.event cimport Event
cimport numpy as np



cdef class Feature:
    """
    This object simply represents a feature, it has an id, a global count, and a value.
    """

    cdef public:
        int id
        int count
        object value
        str parent




cdef class FeatureExtractor:

    cpdef list extract(self, Event event, list features=?)

    cpdef str name(self)


cdef class JumpExtractor(FeatureExtractor):

    cdef:
        str _name



cdef class FeatureMatrix:
    """
    This object holds feature matrices describing each generating context (English words).
    Each feature matrix describes decisions (French words) in terms of sparse features.
    """

    cdef:
        list _matrices
        list _reversed_index
        size_t _d

    cpdef object sparse_zero_vec(self)

    cpdef np.float_t[::1] dense_zero_vec(self)

    cpdef size_t dimensionality(self)

    cpdef object feature_matrix(self, int context)

    cpdef object feature_vector(self, int context, int decision)

    cpdef Feature raw_feature_value(self, size_t column)


cdef np.float_t[:,::1] make_cpds(np.float_t[::1] weight_vector,
                                 FeatureMatrix feature_matrix,
                                 size_t n_contexts,
                                 size_t n_decisions)


cpdef FeatureMatrix make_feature_matrices(GenerativeComponent component,
                                          Corpus e_corpus,
                                          Corpus f_corpus,
                                          extractors,
                                          dict min_occurrences=?,
                                          dict max_occurrences=?)
