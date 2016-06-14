cimport numpy as np
import numpy as np
from lola.feature_vector import FeatureMatrix
import logging
cimport cython


cdef class LogisticRegression:
    """"
    This class deals with the computation of categorical parameters
    based on a log-linear formulation.

    It is used to produce categorical probabilities as a function of
    a parameter vector (w).
    """

    def __init__(self, feature_matrix,
                 np.float_t[::1] weight_vector,
                 int e_vocab_size,
                 int f_vocab_size):
        self._feature_matrix = feature_matrix
        self._weight_vector = weight_vector
        self._e_vocab_size = e_vocab_size
        self._f_vocab_size = f_vocab_size
        # we initialise a cache with negative values for numerators
        # where a negative value means that the potential associated with a certain pair (e,f) has not
        # yet been computed
        self._numerator_cache = np.full((e_vocab_size, f_vocab_size), -1, dtype=float)
        # we initialise a cache with negative values for the denominators
        # denominators are sums over potentials, thus they can never be negative
        # here a negative value simply indicates we haven't yet computed Z(e) for a given e.
        self._denominator_cache = np.full(e_vocab_size, -1, dtype=float)

    @cython.cdivision(True)
    cpdef float probability(self, int e, int f):
        """
        Because we don't have the e and f sentences, we need another formula to calculate theta,
        also because I have not saved the theta's somewhere on forehand
        returns theta_c,d(w) = exp(<w,f(c,d )>)/sum_d'(exp(<w,f(c,d)>))
        :param e: English word
        :param f: French word
        :return: theta associated with e-f pair
        """
        return self.potential(e, f) / self.denominator(e)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef float _potential(self, int e, int f):
        """
        Compute the potential associated with e and f.
        This version is not lazy, that is, it never checks cache.

        :param e:
        :param f:
        :return:
        """
        phi = self._feature_matrix.get_feature_vector(f, e)
        return np.exp(phi.dot(self._weight_vector)[0])  # phi.dot(w) returns a list/array with one element.

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef float potential(self, int e, int f):
        """
        Calculate the potential (an unnormalised probability) for a given f and e
        :param e: English word
        :param f: French word
        :return: exp(w dot phi)
        """

        cdef:
            float potential = self._numerator_cache[e, f]
        if potential < 0:  # potentials can never be negative, thus we haven't computed this one yet
            # compute the potential
            potential = self._potential(e, f)
            # store it in the cache
            self._numerator_cache[e, f] = potential
        return potential

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef float denominator(self, int e):
        """
        Calculate a denominator Z for a given e, where
            Z = sum_f exp(w * phi(e,f))
        We also cache this computation.

        :param e: English word
        :return: Z
        """
        cdef float Z = self._denominator_cache[e]
        if Z < 0:  # this denominator has not yet been computed
            Z = 0.0
            for f in range(self._f_vocab_size):
                Z += self.potential(e, f)
            self._denominator_cache[e] = Z
        return Z