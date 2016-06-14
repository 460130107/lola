"""
:Authors: - Wilker Aziz
"""

import lola.sparse as sparse
import numpy as np
from lola.feature_vector import FeatureMatrix
from scipy.sparse import csr_matrix
import logging
from lola.logreg import LogisticRegression


class ObjectiveAndGradient:

    def __init__(self, expected_counts: sparse.CPDTable,
                 feature_matrix: FeatureMatrix,
                 e_vocab_size: int,
                 f_vocab_size: int):
        """

        :param expected_counts: this is the result of the E-step (that is, the
            part that remains unchanged during the M-step)
        """

        self._expected_counts = expected_counts
        self._feature_matrix = feature_matrix
        self._e_vocab_size = e_vocab_size
        self._f_vocab_size = f_vocab_size

    def expected_feature_vector(self, e: int, logistic_regression: LogisticRegression) -> csr_matrix:
        """
        calculates expected feature vectors: mu_c = sum_d'(theta(c,d')*f(c,d')) for each context e
        :return: dictionary of expected feature vectors for each context
        """
        mu = self._feature_matrix.sparse_zero_vec()
        # loop over all decisions f and sum (theta * feature_vector)
        for f in range(self._f_vocab_size):
            theta = logistic_regression.probability(e, f)
            feature_vector = self._feature_matrix.get_feature_vector(f, e)
            mu += theta * feature_vector
        return mu

    def evaluate(self, weight_vector: np.array, regulariser_strength=0.0) -> tuple:
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

        logistic_regression = LogisticRegression(self._feature_matrix,
                                                 weight_vector,
                                                 self._e_vocab_size,
                                                 self._f_vocab_size)
        gradient = self._feature_matrix.sparse_zero_vec()
        objective = 0.0  # this is the expected log likelihood
        for e in range(self._e_vocab_size):
            logging.debug('Computing expected features for e=%d', e)
            expected_feature_vector = self.expected_feature_vector(e, logistic_regression)
            logging.debug(' updating gradient')
            for f in range(self._f_vocab_size):
                expected_counts = self._expected_counts.get(e, f)  # expected counts from e-step
                theta = logistic_regression.probability(e, f)  # theta_c,d(w) for each w (thus theta is a vector w's)
                feature_vector = self._feature_matrix.get_feature_vector(f, e)
                gradient += expected_counts * (feature_vector - expected_feature_vector)
                objective += expected_counts * np.log(theta)
        # if regulariser_strength != 0.0:
        #   objective -= regulariser_strength * squared(l2_norm(w))
        #   gradient -= 2 * regulariser_strength * w
        return objective, gradient
