from libc.math cimport floor


cdef class EventSpace:

    def __init__(self, shape):
        self.shape = tuple(shape)

    cpdef tuple get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        raise NotImplementedError()

    cpdef tuple readable(self, tuple event):
        return event


cdef class DummyEventSpace(EventSpace):

    def __init__(self):
        super(DummyEventSpace, self).__init__(tuple())

    cpdef tuple get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        return tuple()


cdef class LexEventSpace(EventSpace):

    def __init__(self, Corpus e_corpus, Corpus f_corpus):
        super(LexEventSpace, self).__init__((e_corpus.vocab_size(), f_corpus.vocab_size()))
        self._e_corpus = e_corpus
        self._f_corpus = f_corpus

    cpdef tuple get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        return e_snt[i], f_snt[j]

    cpdef tuple readable(self, tuple event):
        return self._e_corpus.translate(event[0]), self._f_corpus.translate(event[1])


cdef class JumpEventSpace(EventSpace):
    """
    This reserves mass for everything from -L to L-1 passing by 0,
    where L is the length of the longest English sentence.
    """

    def __init__(self, int longest):
        super(JumpEventSpace, self).__init__((1, 2 * longest))
        self._longest = longest

    cpdef tuple get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        cdef:
            int l = e_snt.shape[0]
            int m = f_snt.shape[0]
            int jump = i - <int>floor((j + 1) * float(l) / m)
        if jump < -self._longest:
            jump = - self._longest
        elif jump >= self._longest:
            jump = self._longest - 1
        cdef size_t d = self._longest + jump
        return 0, d

    cpdef tuple readable(self, tuple event):
        return 0, event[1] - self._longest