"""
:Authors: - Wilker Aziz
"""

import numpy as np
cimport numpy as np
import scipy as sp
from lola.frepr cimport LexicalFeatureMatrix
from lola.logreg cimport LogisticRegression
from scipy.sparse import csr_matrix


cdef class ObjectiveAndGradient:

    cdef:
        LexicalFeatureMatrix _feature_matrix
        size_t _e_vocab_size
        size_t _f_vocab_size
        object _sparse_expected_counts

    def __init__(self, sparse_counts,
                 LexicalFeatureMatrix feature_matrix,
                 size_t e_vocab_size,
                 size_t f_vocab_size):
        """

        :param sparse_counts: this is the result of the E-step (that is, the
            part that remains unchanged during the M-step)
        """
        self._feature_matrix = feature_matrix
        self._e_vocab_size = e_vocab_size
        self._f_vocab_size = f_vocab_size
        self._sparse_expected_counts = csr_matrix(sparse_counts)

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
            update = self._sparse_expected_counts[e].dot(self._feature_matrix.feature_matrix(e)) - mu_e * self._sparse_expected_counts[e].sum()

            gradient += update

            # This is the expected log-likelihood
            objective += self._sparse_expected_counts[e].dot(np.log(logistic_regression.categorical(e)))[0]  # matrix (1,) -> scalar

        # if regulariser_strength != 0.0:
        #   objective -= regulariser_strength * squared(l2_norm(w))
        #   gradient -= 2 * regulariser_strength * w

        return objective, gradient.A[0]  # matrix to standard np.array
