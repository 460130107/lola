cimport numpy as np


cdef class Corpus:
    """
    A corpus is a collection of sentences.
    Each sentence is a sequence of words.

    Internally, words are represented as integers for compactness and quick indexing using numpy arrays.

    Remark: This object offers no guarantee as to which exact index any word will get. Not even the NULL word.
    """

    cdef object _lookup
    cdef np.int_t[::1] _inverse
    cdef np.int_t[::1] _boundaries

    cpdef np.int_t[::1] sentence(self, size_t i)

    cpdef translate(self, size_t i)

    cpdef size_t vocab_size(self)

    cpdef size_t corpus_size(self)

    cpdef size_t n_sentences(self)
