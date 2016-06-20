from libc.math cimport floor


cdef class Context:

    def __init__(self, size_t id):
        self.id = id


cdef class Decision:

    def __init__(self, size_t id):
        self.id = id


cdef class Event:

    def __init__(self, Context context, Decision decision):
        self.context = context
        self.decision = decision


cdef class EventSpace:

    cpdef Event get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        return Event(Context(0), Decision(0))

    cpdef size_t n_contexts(self):
        return 1

    cpdef size_t n_decisions(self):
        return 1

    cpdef tuple shape(self):
        return self.n_contexts(), self.n_decisions()


cdef class LexContext(Context):

    def __init__(self, int e_word):
        super(LexContext, self).__init__(e_word)

    cpdef int word(self):
        return self.id


cdef class LexDecision(Decision):

    def __init__(self, int f_word):
        super(LexDecision, self).__init__(f_word)

    cpdef int word(self):
        return self.id


cdef class LexEventSpace(EventSpace):

    def __init__(self, size_t e_vocab_size, size_t f_vocab_size):
        cdef int e, f
        self._contexts = [LexContext(e) for e in range(e_vocab_size)]
        self._decisions = [LexDecision(f) for f in range(f_vocab_size)]

    cpdef Event get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        return Event(LexContext(e_snt[i]), LexDecision(f_snt[j]))

    cpdef size_t n_contexts(self):
        return len(self._contexts)

    cpdef size_t n_decisions(self):
        return len(self._decisions)


cdef class JumpContext(Context):

    def __init__(self):
        super(JumpContext, self).__init__(0)


cdef class JumpDecision(Decision):

    def __init__(self, size_t id, int jump):
        super(JumpDecision, self).__init__(id)
        self._jump = jump

    cpdef int jump(self):
        return self._jump


cdef class JumpEventSpace(EventSpace):

    def __init__(self, max_english_length):
        self._context = JumpContext()
        self._decisions = [JumpDecision(n, i) for n, i in enumerate(range(-max_english_length, max_english_length + 1))]

    cpdef Event get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        cdef:
            size_t l = e_snt.shape[0]
            size_t m = f_snt.shape[0]
            int jump = i - <int>floor(float(j * l) / m)
        return Event(self._context, self._decisions[jump])

    cpdef size_t n_contexts(self):
        return 1

    cpdef size_t n_decisions(self):
        return len(self._decisions)


cdef class DistContext(Context):

    def __init__(self, size_t id, int j, int l, int m):
        super(DistContext, self).__init__(id)
        self.j = j
        self.l = l
        self.m = m


cdef class DistDecision(Decision):

    def __init__(self, int i):
        super(DistDecision, self).__init__(i)

    cpdef int i(self):
        return self.id


cdef class DistEventSpace(EventSpace):

    def __init__(self, size_t max_english_length):
        self._contexts = {}
        self._decisions = [DistDecision(i) for i in range(max_english_length + 1)]

    cpdef Event get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        cdef:
            size_t l = e_snt.shape[0]
            size_t m = f_snt.shape[0]
            tuple ctxt_key = (j, l, m)
        cdef DistContext context = self._contexts.get(ctxt_key, None)
        if context is None:
            context = DistContext(len(self._contexts), j, l, m)
            self._contexts[ctxt_key] = context
        cdef DistDecision decision = self._decisions[i]
        return Event(context, decision)

    cpdef size_t n_contexts(self):
        return len(self._contexts)

    cpdef size_t n_decisions(self):
        return len(self._decisions)

