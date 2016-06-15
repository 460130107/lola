"""
A module with efficient feature representations.

"""

from lola.corpus cimport Corpus
from scipy import sparse
cimport numpy as np


cdef class LexicalFeatureMatrix:
    """
    This object holds feature matrices describing each generating context (English words).
    Each feature matrix describes decisions (French words) in terms of sparse features.
    """

    cdef:
        list _matrices  # one csr_matrix per English word
        list _reversed_index
        size_t _d  # dimensionality of feature space

    cpdef object sparse_zero_vec(self)

    cpdef np.float_t[::1] dense_zero_vec(self)

    cpdef size_t dimensionality(self)

    cpdef object feature_matrix(self, int e)

    cpdef object feature_vector(self, int e, int f)

    cpdef object raw_feature_value(self, size_t column)


cdef class Feature:
    """
    This object simply represents a feature, it has an id, a global count, and a value.
    """

    cdef public:
        int id
        int count
        object value


cpdef LexicalFeatureMatrix make_lexical_matrices(Corpus e_corpus,
                                                 Corpus f_corpus,
                                                 extractors,
                                                 int min_occurrences=?,
                                                 int max_occurrences=?)