"""
Generative components for alignment models.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc cimport math as c_math

cdef class GenerativeComponent:

    def __init__(self, str name):
        self._name = name

    cpdef str name(self):
        return self._name

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """Get the component value associated with a decision a_j=i."""
        pass

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """Adds to the component value associated with a decision a_j=i"""
        pass

    cpdef normalise(self):
        """Normalise the generative component"""
        pass

    cpdef GenerativeComponent zeros(self):
        """
        Return a 0-counts version of the generative component.
        This is useful in gathering sufficient statistics.
        """
        pass

    cpdef save(self, Corpus e_corpus, Corpus f_corpus, str path):
        pass


cdef float cmp_prob(tuple pair):
    return -pair[1]

cdef class LexicalParameters(GenerativeComponent):
    """
    This is a collection of sparse categorical distributions:
        * one distribution per English word
        * each distribution defined over the French vocabulary
    """

    def __init__(self, size_t e_vocab_size, size_t f_vocab_size, float p=0.0, str name='lexical'):
        """

        :param e_vocab_size: size of English vocabulary (number of categorical distributions)
        :param f_vocab_size: size of French vocabulary (support of each categorical distribution)
        :param p: initial value (e.g. use 1.0/f_vocab_size to get uniform distributions)
        """
        super(LexicalParameters, self).__init__(name)
        self._e_vocab_size = e_vocab_size
        self._f_vocab_size = f_vocab_size
        self._cpds = CPDTable(e_vocab_size, f_vocab_size, p)

    cpdef size_t e_vocab_size(self):
        return self._e_vocab_size

    cpdef size_t f_vocab_size(self):
        return self._f_vocab_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """Get the parameter value associated with cat(f|e)."""
        return self._cpds.get(e_snt[i], f_snt[j])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """Adds to the parameter value associated with cat(f|e)."""
        return self._cpds.plus_equals(e_snt[i], f_snt[j], p)

    cpdef normalise(self):
        """Normalise each distribution by its total mass."""
        self._cpds.normalise()

    cpdef GenerativeComponent zeros(self):
        return LexicalParameters(self._e_vocab_size, self._f_vocab_size, 0.0)

    cpdef save(self, Corpus e_corpus, Corpus f_corpus, str path):
        cdef size_t e
        cdef size_t f
        cdef float p
        cdef tuple pair
        with open('{0}.{1}'.format(path, self.name()), 'w') as fo:
            for e in range(self._e_vocab_size):
                for f, p in sorted(self._cpds.iternonzero(e), key=cmp_prob):
                    print('{0} {1} {2}'.format(e_corpus.translate(e), f_corpus.translate(f), p), file=fo)


cdef class DistortionParameters(GenerativeComponent):

    pass

cdef class UniformAlignment(DistortionParameters):

    def __init__(self, str name='uniformdist'):
        super(UniformAlignment, self).__init__(name)

    @cython.cdivision(True)
    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Parameter associated with a certain jump.
        """
        return 1.0 / e_snt.shape[0]

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        return p

    cpdef normalise(self):
        pass

    cpdef GenerativeComponent zeros(self):
        return self


cdef class JumpParameters(DistortionParameters):
    """
    Vogel's style distortion (jump) parameters for IBM2.
    """


    def __init__(self, int max_english_len, int max_french_len, float base_value, str name='jump'):
        super(JumpParameters, self).__init__(name)
        self._max_english_len = max_english_len
        self._max_french_len = max_french_len
        self._categorical = SparseCategorical(max_english_len + max_french_len + 1, base_value)

    def __str__(self):
        return 'max-english-len=%d max-french-len=%d cpd=(%s)' % (self._max_english_len,
                                                                  self._max_french_len,
                                                                  self._categorical)

    @cython.cdivision(True)
    cdef int jump(self, int l, int m, int i, int j):
        """
        Return the relative jump.

        :param l: English sentence length (including NULL)
        :param m: French sentence length
        :param i: 0-based English word position
        :param j: 0-based French word position
        :return: i - floor((j + 1) * l / m)
        """
        return i - <int>c_math.floor((j + 1) * l / m)

    cpdef int max_english_len(self):
        return self._max_english_len

    cpdef int max_french_len(self):
        return self._max_french_len

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Parameter associated with a certain jump.
        """
        # compute the jump
        cdef int jump = self.jump(e_snt.shape[0], f_snt.shape[0], i, j)
        # retrieve the parameter associated with it
        # or a base value in case the jump is not yet mapped
        return self._categorical.get(jump)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        # compute the jump
        cdef int jump = self.jump(e_snt.shape[0], f_snt.shape[0], i, j)
        # accumulate fractional count
        return self._categorical.plus_equals(jump, p)

    cpdef normalise(self):
        # sum the values already mapped
        self._categorical.normalise()

    cpdef GenerativeComponent zeros(self):
        return JumpParameters(self.max_english_len(), self.max_french_len(), 0.0)

    cpdef save(self, Corpus e_corpus, Corpus f_corpus, str path):
        cdef size_t e
        cdef size_t f
        cdef float p
        cdef tuple pair
        with open('{0}.{1}'.format(path, self.name()), 'w') as fo:
            for jump, p in sorted(self._categorical.iternonzero(), key=cmp_prob):
                print('{0} {1}'.format(jump, p), file=fo)


cdef class BrownDistortionParameters(DistortionParameters):

    def __init__(self, int max_english_len, float base_value, str name='browndist'):
        super(BrownDistortionParameters, self).__init__(name)
        self._max_english_len = max_english_len
        self._base_value = base_value
        self._cpds = dict()

    cpdef int max_english_len(self):
        return self._max_english_len

    cpdef float base_value(self):
        return self._base_value

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Parameter associated with a certain jump.
        """
        cdef tuple key = (e_snt.shape[0], f_snt.shape[0], j)
        cdef SparseCategorical cpd
        if key not in self._cpds:
            cpd = SparseCategorical(self._max_english_len, self._base_value)
            self._cpds[key] = cpd
        else:
            cpd = self._cpds[key]
        return cpd.get(i)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        cdef tuple key = (e_snt.shape[0], f_snt.shape[0], j)
        cdef SparseCategorical cpd
        if key not in self._cpds:
            cpd = SparseCategorical(self._max_english_len, self._base_value)
            self._cpds[key] = cpd
        else:
            cpd = self._cpds[key]
        return cpd.plus_equals(i, p)

    cpdef normalise(self):
        cdef tuple ctxt
        cdef SparseCategorical cpd
        for ctxt, cpd in self._cpds.items():
            cpd.normalise()

    cpdef GenerativeComponent zeros(self):
        return BrownDistortionParameters(self._max_english_len, self._base_value)