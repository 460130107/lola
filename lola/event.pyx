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

    def __init__(self, size_t max_english_length):
        super(JumpEventSpace, self).__init__((1, 2 * max_english_length + 1))
        self._shift = max_english_length

    cpdef tuple get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        cdef:
            size_t l = e_snt.shape[0]
            size_t m = f_snt.shape[0]
            int jump = i - <int>floor(float(j * l) / m)
        return 0, self._shift + jump

    cpdef tuple readable(self, tuple event):
        return 0, event[1] - self._shift