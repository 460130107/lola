"""

Not using these cython macros for now:

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""

cimport numpy as np
import numpy as np
cimport cython
from collections import Counter
from lola.ptypes cimport uint_t
import lola.ptypes as ptypes


cdef tuple tokenize(istream, bos=None):
    """
    This method tokenizes an input corpus and returns a stream of tokens.

    :param istream: input stream (e.g. file handler)
    :param bos: this is optional and if set it is added to the beginning of every sentence
    :return: an np.array of tokens, and an np.array of boundary positions
    """
    cdef str line
    cdef list tokens = []
    cdef list boundaries = []
    if bos:
        for line in istream:
            tokens.append(bos)
            tokens.extend(line.split())
            boundaries.append(len(tokens))
    else:
        for line in istream:
            tokens.extend(line.split())
            boundaries.append(len(tokens))
    return np.array(tokens, dtype='U'), np.array(boundaries, dtype=ptypes.uint)


cdef tuple tokenize_and_prune(istream, bos=None,
                              size_t min_count=0, size_t max_count=0,
                              str min_tag='<UNK-MIN>', str max_tag='<UNK-MAX>'):
    """
    This method tokenizes an input corpus and returns a stream of tokens.

    :param istream: input stream (e.g. file handler)
    :param bos: this is optional and if set it is added to the beginning of every sentence
    :return: an np.array of tokens, and an np.array of boundary positions
    """
    cdef str line, token
    cdef list tokens = []
    cdef list boundaries = []
    counter = Counter()

    # tokenize
    for line in istream:
        if bos:
            tokens.append(bos)
        for token in line.split():
            tokens.append(token)
            counter.update([token])
        boundaries.append(len(tokens))

    # prune
    if min_count > 1 or max_count > 1:
        for i in range(len(tokens)):
            token = tokens[i]
            if tokens[i] != bos:
                if 1 <= counter[token] < min_count:
                    tokens[i] = min_tag
                elif 1 <= max_count < counter[token]:
                    tokens[i] = max_tag

    return np.array(tokens, dtype='U'), np.array(boundaries, dtype=ptypes.uint)



cdef class Corpus:
    """
    A corpus is a collection of sentences.
    Each sentence is a sequence of words.

    Internally, words are represented as integers for compactness and quick indexing using numpy arrays.

    Remark: This object offers no guarantee as to which exact index any word will get. Not even the NULL word.
    """

    def __init__(self, istream, null=None, size_t min_count=0, size_t max_count=0):
        """
        Creates a corpus from a text file.
        The corpus is internally represented by a flat numpy array.

        :param istream: an input stream or a path to a file
        :param null: an optional NULL token to be added to the beginning of every sentence
        """

        # read and tokenize the entire corpus
        # and if a null symbol is given, we place it at the beginning of the sentence
        # we also memorise the boundary positions
        if type(istream) is str:  # this is actually a path to a file
            with open(istream, 'r') as fstream:
                tokens, self._boundaries = tokenize_and_prune(fstream, bos=null,
                                                              min_count=min_count,
                                                              max_count=max_count)
        else:
            tokens, self._boundaries = tokenize_and_prune(istream, bos=null,
                                                          min_count=min_count,
                                                          max_count=max_count)
        # use numpy to map tokens to integers
        # lookup converts from integers back to strings
        # inverse represents the corpus with words represented by integers
        self._lookup, inverse = np.unique(tokens, return_inverse=True)
        self._inverse = np.array(inverse, dtype=ptypes.uint)

        cdef int a = 0
        cdef int max_len = 0
        cdef int b
        for b in self._boundaries:
            if b - a > max_len:
                max_len = b - a
            a = b
        self._max_len = <size_t>max_len

    cpdef size_t max_len(self):
        """Returns the length of the longest sentence in the corpus."""
        return self._max_len

    cpdef uint_t[::1] sentence(self, size_t i):
        """
        Return the ith sentence. This is not checked for out-of-bound conditions.
        :param i: 0-based sentence id
        :return: memory view corresponding to the sentence
        """
        cdef uint_t a = 0 if i == 0 else self._boundaries[i - 1]
        cdef uint_t b = self._boundaries[i]
        return self._inverse[a:b]

    def itersentences(self):
        """Iterates over sentences"""
        cdef size_t a = 0
        cdef size_t b
        for b in self._boundaries:
            yield self._inverse[a:b]  # this produces a view, not a copy ;)
            a = b

    cpdef translate(self, size_t i):
        """
        Translate an integer back to a string.
        :param i: index representing the word
        :return: original string
        """
        return self._lookup[i]

    cpdef size_t vocab_size(self):
        """Number of unique tokens (if the corpus was created with added NULL tokens, this will include it)"""
        return len(self._lookup)

    cpdef size_t corpus_size(self):
        """Number of tokens in the corpus."""
        return self._inverse.shape[0]

    cpdef size_t n_sentences(self):
        """Number of sentences in the corpus."""
        return self._boundaries.shape[0]

    cpdef Corpus underlying(self):
        """Returns itself"""
        return self



cdef class CorpusView(Corpus):
    """
    A corpus view wraps a Corpus object exposing only a subset of its sentences.

    A few things remain unchaged though:
        * max length
        * vocab size
        * corpus size (in tokens)
    These are simply delegated to the underlying corpus.

    This class exists to unify training and test vocabulary and word ids.
    """

    def __init__(self, Corpus corpus, size_t offset, size_t size):
        """

        :param corpus: a Corpus
        :param offset: where to start from (0-based)
        :param size: how many sentences to include
        """
        self._corpus = corpus
        self._offset = offset
        self._size = size

    cpdef size_t max_len(self):
        """Returns the length of the longest sentence in the corpus."""
        return self._corpus.max_len()

    cpdef uint_t[::1] sentence(self, size_t i):
        """
        Return the ith sentence. This is not checked for out-of-bound conditions.
        :param i: 0-based sentence id
        :return: memory view corresponding to the sentence
        """
        return self._corpus.sentence(self._offset + i)

    def itersentences(self):
        """Iterates over sentences"""
        cdef size_t i
        for i in range(self._offset, self._offset + self._size):
            yield self._corpus.sentence(i)

    cpdef translate(self, size_t i):
        """
        Translate an integer back to a string.
        :param i: index representing the word
        :return: original string
        """
        return self._corpus.translate(i)

    cpdef size_t vocab_size(self):
        """Number of unique tokens (if the corpus was created with added NULL tokens, this will include it)"""
        return self._corpus.vocab_size()

    cpdef size_t corpus_size(self):
        """Number of tokens in the corpus."""
        return self._corpus.corpus_size()

    cpdef size_t n_sentences(self):
        """Number of sentences in the corpus."""
        return self._size

    cpdef Corpus underlying(self):
        """Returns the underlying Corpus object"""
        return self._corpus

