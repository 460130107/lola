from lola.corpus cimport Corpus
from lola.event cimport EventSpace
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


cdef class DenseFeatureMatrix:
    """
    This object holds feature matrices describing each generating context (English words).
    Each feature matrix describes decisions (French words) in terms of sparse features.
    """

    cdef:
        np.float_t[:,:,::1] _matrices
        list _descriptors
        size_t _d

    cpdef np.float_t[::1] zeros(self)

    cpdef size_t dimensionality(self)

    cpdef object feature_matrix(self, int context)

    cpdef object dots(self, int context, weights)

    cpdef object expected_fvector(self, int context, np.float_t[::1] cpd)

    cpdef object feature_vector(self, int context, int decision)

    cpdef Feature descriptor(self, size_t column)


cdef class EmptyDenseFeatureMatrix(DenseFeatureMatrix):
    pass


cdef class SparseFeatureMatrix:
    """
    This object holds feature matrices describing each generating context (English words).
    Each feature matrix describes decisions (French words) in terms of sparse features.
    """

    cdef:
        object _matrices
        list _reversed_index
        size_t _C, _D, _F

    cpdef object sparse_zero_vec(self)

    cpdef np.float_t[::1] dense_zero_vec(self)

    cpdef size_t dimensionality(self)

    cpdef object feature_matrix(self, int context)

    cpdef object dots(self, int context, np.float_t[::1] weights)

    cpdef object expected_fvector(self, int context, np.float_t[::1] cpd)

    cpdef object feature_vector(self, int context, int decision)

    cpdef Feature raw_feature_value(self, size_t column)


cdef class EmptySparseFeatureMatrix(SparseFeatureMatrix):

    pass


cdef np.float_t[:,::1] make_cpds(np.float_t[::1] weight_vector,
                                 SparseFeatureMatrix feature_matrix,
                                 size_t n_contexts,
                                 size_t n_decisions)


cdef np.float_t[:,::1] make_cpds2(np.float_t[::1] wd,
                                  np.float_t[::1] ws,
                                  DenseFeatureMatrix dense_matrix,
                                  SparseFeatureMatrix sparse_matrix,
                                  size_t n_contexts,
                                  size_t n_decisions)


cpdef SparseFeatureMatrix make_sparse_matrices(EventSpace event_space,
                                          Corpus e_corpus,
                                          Corpus f_corpus,
                                          extractors,
                                          dict min_occurrences=?,
                                          dict max_occurrences=?)

cpdef DenseFeatureMatrix make_dense_matrices(EventSpace event_space,
                                          Corpus e_corpus,
                                          Corpus f_corpus,
                                          extractors)
