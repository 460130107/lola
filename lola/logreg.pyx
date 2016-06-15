cimport numpy as np
import numpy as np


cdef np.float_t[:,::1] make_categoricals(np.float_t[::1] weight_vector,
                                         LexicalFeatureMatrix feature_matrix,
                                         size_t e_vocab_size,
                                         size_t f_vocab_size):
    cdef:
        size_t e
        float total
    w = np.array(weight_vector)
    numerators = np.zeros((e_vocab_size, f_vocab_size), dtype=float)
    denominators = np.zeros(e_vocab_size, dtype=float)
    for e in range(e_vocab_size):
        e_matrix = feature_matrix.feature_matrix(e)
        numerators[e] = np.exp(e_matrix.dot(w))  # this is a sparse dot product ;)
        total = numerators[e].sum()
        denominators[e] = total
    return numerators / denominators[:,np.newaxis]


cdef class LogisticRegression:
    """"
    This class deals with the computation of categorical parameters
    based on a log-linear formulation.

    It is used to produce categorical probabilities as a function of
    a parameter vector (w).
    """

    def __init__(self, LexicalFeatureMatrix feature_matrix,
                 np.float_t[::1] weight_vector,
                 int e_vocab_size,
                 int f_vocab_size):
        self._categoricals = make_categoricals(weight_vector,
                                               feature_matrix,
                                               e_vocab_size,
                                               f_vocab_size)

    cpdef float probability(self, int e, int f):
        """
        Because we don't have the e and f sentences, we need another formula to calculate theta,
        also because I have not saved the theta's somewhere on forehand
        returns theta_c,d(w) = exp(<w,f(c,d )>)/sum_d'(exp(<w,f(c,d)>))
        :param e: English word
        :param f: French word
        :return: theta associated with e-f pair
        """
        return self._categoricals[e, f]

    cpdef np.float_t[::1] categorical(self, int e):
        return self._categoricals[e]
