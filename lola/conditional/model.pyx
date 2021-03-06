import numpy as np
cimport cython


cdef class GenerativeModel:
    """
    A 0th-order alignment model, that is, alignment links are independent on one another.
    """

    def __init__(self, components):
        assert len(components) > 0, 'I need at least one generative component'
        self._components = tuple(components)

    def __iter__(self):
        return iter(self._components)

    cpdef size_t n_components(self):
        return len(self._components)

    cpdef GenerativeComponent component(self, size_t n):
        return self._components[n]

    cpdef real_t likelihood(self, uint_t[::1] e_snt, uint_t[::1] f_snt, size_t i, size_t j):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: p(a_j = i, f | e)
        """
        cdef GenerativeComponent comp
        cdef real_t l = 1.0
        for comp in self._components:
            l *= comp.prob(e_snt, f_snt, i, j)
        return l

    cpdef observe(self, uint_t[::1] e_snt, uint_t[::1] f_snt, size_t i, size_t j, real_t p):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :param p: posterior probability
        """
        cdef GenerativeComponent comp
        for comp in self._components:
            comp.observe(e_snt, f_snt, i, j, p)

    cpdef update(self):
        cdef GenerativeComponent comp
        for comp in self._components:
            comp.update()

    cpdef load(self, path):
        cdef GenerativeComponent comp
        for comp in self._components:
            comp.load('%s.%s' % (path, comp.name))

    cpdef save(self, path):
        cdef GenerativeComponent comp
        for comp in self._components:
            comp.save('%s.%s' % (path, comp.name))