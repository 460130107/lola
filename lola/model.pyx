import numpy as np


cdef class SufficientStatistics:
    """
    This is used to gather sufficient statistics under a certain model.
    The typical use is to accumulated expected counts from (potential) observations.

    This object also knows how to construct a new model based on up-to-date statistics.
    """

    cpdef observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """
        Account for a potential observation.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :param p: probability of observation, i.e., normalised posterior p(a_j=i | f, e)
        """
        pass


cdef class ExpectedCounts(SufficientStatistics):
    """
    This is used to gather sufficient statistics under a certain model.
    The typical use is to accumulated expected counts from (potential) observations.

    This object also knows how to construct a new model based on up-to-date statistics.
    """

    def __init__(self, components):
        self._components = list(components)

    cpdef observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """
        Account for a potential observation.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :param p: probability of observation, i.e., normalised posterior p(a_j=i | f, e)
        """
        cdef GenerativeComponent comp
        for comp in self._components:
            comp.plus_equals(e_snt, f_snt, i, j, p)

    cpdef list components(self):
        return self._components


cdef class Model:
    """
    A 0th-order alignment model, that is, alignment links are independent on one another.
    """

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: p(a_j = i, f | e)
        """
        pass

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i | f, e) up to a normalisation constant.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: Z * p(a_j = i | f, e)
        """
        pass

    cpdef initialise(self, dict initialiser):
        """
        Some models can be initialised from other models parameters.
        """
        pass

    cpdef SufficientStatistics suffstats(self):
        """
        Return an object that gather sufficient statistics (to be used in the E-step).
        """
        pass

    cpdef update(self, list components):
        """
        Make a new instance of the same model based on gathered sufficient statistics (to be used in the M-step).
        """
        pass


cdef class GenerativeModel(Model):
    """
    A 0th-order alignment model, that is, alignment links are independent on one another.
    """

    def __init__(self, components):
        self._components = list(components)

    def __iter__(self):
        return iter(self._components)

    cpdef size_t n_components(self):
        return len(self._components)

    cpdef GenerativeComponent component(self, int i):
        return self._components[i]

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: p(a_j = i, f | e)
        """
        cdef GenerativeComponent comp
        cdef float l = 1.0
        for comp in self._components:
            l *= comp.get(e_snt, f_snt, i, j)
        return l

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i | f, e) up to a normalisation constant.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: Z * p(a_j = i | f, e)
        """
        cdef GenerativeComponent comp
        cdef float l = 1.0
        for comp in self._components:
            l *= comp.get(e_snt, f_snt, i, j)
        return l

    cpdef SufficientStatistics suffstats(self):
        """
        This is a collection of counters for generative components.
        :return:
        """
        cdef list components = []
        cdef GenerativeComponent comp
        for comp in self._components:
            components.append(comp.zeros())
        return ExpectedCounts(components)

    cpdef update(self, list components):
        self._components = list(components)
