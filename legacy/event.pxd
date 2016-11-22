cimport numpy as np


cdef class Context:

    cdef readonly size_t id


cdef class Decision:

    cdef readonly size_t id


cdef class Event:

    cdef readonly:
        Context context
        Decision decision


cdef class EventSpace:

    cpdef Event get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j)

    cpdef Event fetch(self, int context_id, int decision_id)

    cpdef size_t n_contexts(self)

    cpdef size_t n_decisions(self)

    cpdef tuple shape(self)


cdef class LexContext(Context):

    cpdef int word(self)


cdef class LexDecision(Decision):

    cpdef int word(self)


cdef class LexEvent(Event):

    pass


cdef class LexEventSpace(EventSpace):

    cdef:
        list _contexts
        list _decisions


cdef class JumpContext(Context):

    pass


cdef class JumpDecision(Decision):

    cdef int _jump

    cpdef int jump(self)


cdef class JumpEvent(Event):

    pass


cdef class JumpEventSpace(EventSpace):

    cdef:
        JumpContext _context
        list _decisions
        size_t _shift


cdef class DistContext(Context):

    cdef readonly int j, l, m


cdef class DistDecision(Decision):

    cpdef int i(self)


cdef class DistEvent(Event):

    pass


cdef class DistEventSpace(EventSpace):

    cdef:
        dict _contexts
        list _decisions

