from legacy.corpus cimport Corpus
from legacy.event cimport Event
import re
cimport numpy as np


cdef class FeatureExtractor:

    cdef str _name

    cpdef list extract(self, Event event)

    cpdef list extract_dense(self, Event event)

    cpdef size_t n_dense(self)

    cpdef list dense_names(self)

    cpdef str name(self)


cdef class LexicalFeatureExtractor(FeatureExtractor):

    cdef:
        Corpus e_corpus
        Corpus f_corpus
        bint extract_e
        bint extract_f
        bint extract_ef


cdef class IBM1Probabilities(LexicalFeatureExtractor):

   cdef np.float_t[:,::1] _ibm1_prob


cdef class WholeWordFeatureExtractor(LexicalFeatureExtractor):
    """
    Example class using the word itself as feature only (in both English as French)
    """

    pass

cdef class AffixFeatureExtractor(LexicalFeatureExtractor):

    cdef:
        list suffix_sizes
        list prefix_sizes
        size_t min_e_length
        size_t min_f_length


cdef class CategoryFeatureExtractor(LexicalFeatureExtractor):

    cdef:
        object digits_re


cdef class LengthFeatures(LexicalFeatureExtractor):

    cdef:
        size_t _n_dense
        list _dense_names


cdef class JumpFeatureExtractor(FeatureExtractor):

    cdef:
        list _bins


cdef class DistortionFeatureExtractor(FeatureExtractor):

    pass
