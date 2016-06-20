from lola.corpus cimport Corpus
from lola.event cimport Event
import re


cdef class FeatureExtractor:

    cdef str _name

    cpdef list extract(self, Event event, list features=?)

    cpdef str name(self)


cdef class LexicalFeatureExtractor(FeatureExtractor):

    cdef:
        Corpus e_corpus
        Corpus f_corpus
        bint extract_e
        bint extract_f
        bint extract_ef


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


cdef class JumpFeatureExtractor(FeatureExtractor):

    cdef:
        list _bins


cdef class DistortionFeatureExtractor(FeatureExtractor):

    pass
