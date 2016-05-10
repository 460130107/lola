"""
Generative components for alignment models.
"""
import numpy as np
cimport numpy as np


cdef class GenerativeComponent:

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


cdef class LexicalParameters(GenerativeComponent):
    """
    This is a collection of sparse categorical distributions:
        * one distribution per English word
        * each distribution defined over the French vocabulary
    """

    def __init__(self, int e_vocab_size, int f_vocab_size, float p=0.0):
        """

        :param e_vocab_size: size of English vocabulary (number of categorical distributions)
        :param f_vocab_size: size of French vocabulary (support of each categorical distribution)
        :param p: initial value (e.g. use 1.0/f_vocab_size to get uniform distributions)
        """
        self._cpds = [SparseCategorical(f_vocab_size, p) for _ in range(e_vocab_size)]

    def __str__(self):
        cdef int i
        cdef list lines = []
        cdef SparseCategorical cpd
        for i, cpd in enumerate(self._cpds):
            lines.append('%d: %s' % (i, str(cpd)))
        return '\n'.join(lines)

    cpdef size_t e_vocab_size(self):
        return len(self._cpds)

    cpdef size_t f_vocab_size(self):
        return self._cpds[0].support_size() if len(self._cpds) else 0

    cpdef SparseCategorical row(self, int e):
        """Return the categorical associated with the conditioning context e."""
        return self._cpds[e]

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """Get the parameter value associated with cat(f|e)."""
        return self._cpds[e_snt[i]].get(f_snt[j])

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """Adds to the parameter value associated with cat(f|e)."""
        return self._cpds[e_snt[i]].plus_equals(f_snt[j], p)

    cpdef normalise(self):
        """Normalise each distribution by its total mass."""
        cdef SparseCategorical cpd
        for cpd in self._cpds:
            cpd.normalise()

    cpdef GenerativeComponent zeros(self):
        return LexicalParameters(self.e_vocab_size(), self.f_vocab_size(), 0.0)


cdef class UniformAlignment(GenerativeComponent):

    def __init__(self):
        pass

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Parameter associated with a certain jump.
        """
        return 1.0 / len(e_snt)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        return p

    cpdef normalise(self):
        pass

    cpdef GenerativeComponent zeros(self):
        return UniformAlignment()


cdef class JumpParameters(GenerativeComponent):
    """
    Vogel's style distortion (jump) parameters for IBM2.
    """


    def __init__(self, int max_english_len, int max_french_len, float base_value):
        self._max_english_len = max_english_len
        self._max_french_len = max_french_len
        self._categorical = SparseCategorical(max_english_len + max_french_len + 1, base_value)

    def __str__(self):
        return 'max-english-len=%d max-french-len=%d cpd=(%s)' % (self._max_english_len,
                                                                  self._max_french_len,
                                                                  self._categorical)

    cpdef int jump(self, int l, int m, int i, int j):
        """
        Return the relative jump.

        :param l: English sentence length (including NULL)
        :param m: French sentence length
        :param i: 0-based English word position
        :param j: 0-based French word position
        :return: i - floor((j + 1) * l / m)
        """
        return i - np.floor((j + 1) * l / m)

    cpdef int max_english_len(self):
        return self._max_english_len

    cpdef int max_french_len(self):
        return self._max_french_len

    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Parameter associated with a certain jump.
        """
        # compute the jump
        cdef int jump = self.jump(len(e_snt), len(f_snt), i, j)
        # retrieve the parameter associated with it
        # or a base value in case the jump is not yet mapped
        return self._categorical.get(jump)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        # compute the jump
        cdef int jump = self.jump(len(e_snt), len(f_snt), i, j)
        # accumulate fractional count
        return self._categorical.plus_equals(jump, p)

    cpdef normalise(self):
        # sum the values already mapped
        self._categorical.normalise()

    cpdef GenerativeComponent zeros(self):
        return JumpParameters(self.max_english_len(), self.max_french_len(), 0.0)


cdef class BrownDistortionParameters(GenerativeComponent):

    def __init__(self, int max_english_len, float base_value):
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
        cdef tuple key = (len(e_snt), len(f_snt), j)
        cdef SparseCategorical cpd
        if key not in self._cpds:
            cpd = SparseCategorical(self._max_english_len, self._base_value)
            self._cpds[key] = cpd
        else:
            cpd = self._cpds[key]
        return cpd.get(i)

    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        cdef tuple key = (len(e_snt), len(f_snt), j)
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