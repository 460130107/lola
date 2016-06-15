from lola.corpus cimport Corpus
import re


cdef class LexicalFeatureExtractor:

    cdef:
        Corpus e_corpus
        Corpus f_corpus
        bint extract_e
        bint extract_f
        bint extract_ef

    cpdef list extract(self, int e, int f, list features=?)


cpdef list extract_lexical_features(int e, int f, list lexical_extractors)


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
