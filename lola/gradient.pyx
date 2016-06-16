"""
:Authors: - Wilker Aziz
"""

import numpy as np
cimport numpy as np
import scipy as sp
from lola.sparse cimport CPDTable
from lola.frepr cimport LexicalFeatureMatrix
from lola.logreg cimport LogisticRegression

from scipy.sparse import csr_matrix
import logging

from time import time


cpdef object csr_expected_difference(matrix, row, probs):
    cdef:
        size_t n_rows = matrix.shape[0]
        size_t n_cols = matrix.shape[1]
        size_t i
        np.float_t[::1] m_probs = probs
    u = csr_matrix((1, n_cols))
    for i in range(n_rows):
        u += (matrix[i] - row) * m_probs[i]
    return u


cpdef object dense_expected_difference(matrix, row, probs):
    cdef:
        size_t n_rows = matrix.shape[0]
        size_t n_cols = matrix.shape[1]
        size_t i
        np.float_t[::1] m_probs = probs
        np.float_t[::1] drow = row.A[0]
    u = probs.dot(matrix - row.todense())
    #u = np.zeros(n_cols)
    #for i in range(n_rows):
    #    u += (matrix[i].A[0] - drow) * m_probs[i]
    return u


cdef class ObjectiveAndGradient:

    cdef:
        CPDTable _expected_counts
        LexicalFeatureMatrix _feature_matrix
        size_t _e_vocab_size
        size_t _f_vocab_size
        object _sparse_counts

    def __init__(self, CPDTable expected_counts, sparse_counts,
                 LexicalFeatureMatrix feature_matrix,
                 size_t e_vocab_size,
                 size_t f_vocab_size):
        """

        :param expected_counts: this is the result of the E-step (that is, the
            part that remains unchanged during the M-step)
        """

        self._expected_counts = expected_counts
        self._feature_matrix = feature_matrix
        self._e_vocab_size = e_vocab_size
        self._f_vocab_size = f_vocab_size
        self._sparse_counts = csr_matrix(sparse_counts)

    cdef expected_feature_vector(self, int e, cpd):
        """
        calculates expected feature vectors: mu_c = sum_d'(theta(c,d')*f(c,d')) for each context e
        :return: dictionary of expected feature vectors for each context
        """
        #mu = self._feature_matrix.sparse_zero_vec()
        e_matrix = self._feature_matrix.feature_matrix(e)  # each row is the feature representation of a French word
        # mu = F' dot P
        return cpd * e_matrix

    cpdef evaluate(self, weight_vector, regulariser_strength=0.0):
        """
            1. Initialise a logistic regression with the new weight vector (w)
                This basically produces a conditional probability for each
                    context (e) and decision (f)
                as a function of w (see second formula on second column in section 3.1 of Kirkpatrick)
            2. Compute expected feature vectors (under the current logistic regression)
                see formula 4 , second line, last term
            3. Gradient and Expected likelihood (which reuses the fixed expected counts)
                see formula 3 and 4
        :returns: expected log likelihood (for maximisation) and gradient (csr_matrix)
        """

        cdef LogisticRegression logistic_regression = LogisticRegression(self._feature_matrix,
                                                 weight_vector,
                                                 self._e_vocab_size,
                                                 self._f_vocab_size)
        logging.debug('Preprocessing CPDs')
        # TODO: can be faster
        expected_counts = np.array([np.array([self._expected_counts.get(e, f)
                                              for f in range(self._f_vocab_size)])
                                    for e in range(self._e_vocab_size)])
        logging.debug('Computing gradient')

        gradient = sp.matrix(np.zeros(self._feature_matrix.dimensionality()))
        objective = 0.0

        for e in range(self._e_vocab_size):

            # The gradient is
            #   (1) \sum_f <n(e,f)> * Delta(e,f,w)
            # where
            #   <n(e,f> is the expected counts of (e,f) from the E-step
            #   Delta(e,f,w) = phi(e,f) - <phi(e)>
            #   where
            #     phi(e,f) is a feature vector describing (e,f)
            #     and <phi(e)> is the expected feature vector under current w
            #     (2) <phi(e)> = \sum_f theta(e,f,w) * phi(e,f)
            # Now with a bit of algebra we see that we can compute this gradient rather efficiently
            #   First note that with respect to equation (1), <phi(e)> as defined in (2) is a constant,
            #    let us call it \mu.
            #   Now equation (1) becomes,
            #      \sum_f <n(e,f)> * ( phi(e,f) - u) =
            #      (\sum_f <n(e,f)> * phi(e,f)) - u * (\sum_f <n(e,f)>)            (3)
            #   Note that,
            #    i. u is an expectation, typically a dense vector (since it sums over all f-words, see equation 2)
            #    ii. let the vector n =[<n(e,f_1)>,<n(e,f_2)>,...,<n(e,f_{V_F})>] represent the vector of expected counts
            #        this vector is typically sparse, because not many events are expected to co-occur
            #    iii. phi(e,f) is rather sparse, but the result of (\sum_f <n(e,f)> * phi(e,f)) is expected to be dense
            #   Thus, in terms of scipy operations, it is convenient to use expression (3),
            #     the first term can be computed efficiently as a dot product between a sparse csr_matrix and
            #     a (sparse or dense) vector;
            #     the second term is just the elementwise product between two vectors.
            #   Using (3) instead of (1) is 3 to 4 orders of magnitude faster!

            # First we compute the expected feature vector of a given English word
            mu_e = self._feature_matrix.feature_matrix(e).T * logistic_regression.categorical(e)

            # super slow version
            # slow_update = expected_counts[e].dot(self._feature_matrix.feature_matrix(e) - d_expected_feature_vector)

            # much faster version
            #t0 = time()
            update = self._feature_matrix.feature_matrix(e).T * expected_counts[e] - mu_e * expected_counts[e].sum()
            #print('update1', time() - t0)

            # Consider this with sparse counts
            #t0 = time()
            #update2 = self._sparse_counts[e].dot(self._feature_matrix.feature_matrix(e)) - mu_e * self._sparse_counts[e].sum()
            #print('update2', time() - t0)


            #for a, b in zip(update, update2):
            #    if not np.allclose([a], [b]):
            #        print('diff', a,b, a-b)
            #assert np.allclose(update, update2.A[0]), 'Oops'

            gradient += update

            # This is the expected log-likelihood
            objective += np.dot(expected_counts[e], np.log(logistic_regression.categorical(e)))

        # if regulariser_strength != 0.0:
        #   objective -= regulariser_strength * squared(l2_norm(w))
        #   gradient -= 2 * regulariser_strength * w

        return objective, gradient
