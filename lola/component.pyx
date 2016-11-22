"""
Generative components for alignment models.
"""

import numpy as np
cimport numpy as np
cimport cython
from lola.event cimport DummyEventSpace
from lola.event cimport LexEventSpace
from lola.event cimport JumpEventSpace
from lola.corpus cimport Corpus


cdef class GenerativeComponent:

    def __init__(self, str name, EventSpace event_space):
        self.name = name
        self.event_space = event_space

    cpdef float prob(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """Get the component value associated with a decision a_j=i."""
        raise NotImplementedError()

    cpdef observe(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """Adds to the component value associated with a decision a_j=i"""
        raise NotImplementedError()

    cpdef update(self):
        """Updates the component (M-step)"""
        raise NotImplementedError()

    cpdef load(self, path):
        pass

    cpdef save(self, path):
        pass


cdef class UniformAlignment(GenerativeComponent):

    def __init__(self, str name='uniform'):
        super(UniformAlignment, self).__init__(name, DummyEventSpace())

    @cython.cdivision(True)
    cpdef float prob(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Parameter associated with a certain jump.
        """
        return 1.0 / e_snt.shape[0]

    cpdef observe(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        pass

    cpdef update(self):
        pass  # there is nothing to update here


cpdef float cmp_prob(tuple pair):
    return -pair[1]


cdef class CategoricalComponent(GenerativeComponent):
    """
    This is a collection of sparse categorical distributions:
        * one distribution per English word
        * each distribution defined over the French vocabulary
    """

    def __init__(self, str name, EventSpace event_space):
        super(CategoricalComponent, self).__init__(name, event_space)
        if len(event_space.shape) > 2:
            raise ValueError("I do not support tensors")
        cdef size_t d1, d2
        d1, d2 = event_space.shape
        # uniform initialisation
        # C++ sparse CPDs
        # self._cpds = np.ones(shape=event_space.shape, dtype=float) / np.prod(event_space.shape)
        self._cpds = CPDTable(d1, d2, 1.0 / d2)
        self._counts = CPDTable(d1, d2, 0.0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float prob(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """Get the parameter value associated with cat(f|e)."""
        cdef size_t c, d
        c, d = self.event_space.get(e_snt, f_snt, i, j)
        return self._cpds.get(c, d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef observe(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """Adds to the parameter value associated with cat(f|e)."""
        cdef size_t c, d
        c, d = self.event_space.get(e_snt, f_snt, i, j)
        self._counts.plus_equals(c, d, p)

    cpdef update(self):
        """CPDs are updated with normalised expected counts, and counts are set to zero."""
        cdef size_t d1, d2
        d1, d2 = self.event_space.shape
        self._counts.normalise()
        self._cpds = self._counts
        self._counts = CPDTable(d1, d2, 0.0)


cdef class BrownLexical(CategoricalComponent):
    """
    Brown's categorical translation distribution.
    """

    def __init__(self, Corpus e_corpus, Corpus f_corpus, str name='lexical'):
        super(BrownLexical, self).__init__(name, LexEventSpace(e_corpus, f_corpus))

    cpdef save(self, path):
        with open(path, 'w') as fo:
            for e in range(len(self._cpds)):
                for f, p in sorted(self._cpds.iternonzero(e), key=cmp_prob):
                    e_str, f_str = self.event_space.readable((e, f))
                    print('%s %s %r' % (e_str, f_str, p), file=fo)

cdef class VogelJump(CategoricalComponent):
    """
    Vogel's style distortion (jump) parameters for IBM2.
    """

    def __init__(self, int max_english_len, str name='jump'):
        super(VogelJump, self).__init__(name, JumpEventSpace(max_english_len))

    cpdef save(self, path):
        with open(path, 'w') as fo:
            for shiftted_jump, p in sorted(self._cpds.iternonzero(0), key=cmp_prob):
                _, jump = self.event_space.readable((0, shiftted_jump))
                print('%d %r' % (jump, p), file=fo)