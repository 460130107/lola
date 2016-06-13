from lola.component import GenerativeComponent
import lola.sparse as sparse
import numpy as np
import collections
from lola.feature_vector import FeatureMatrix
from lola.corpus import Corpus
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import logging
from lola.logreg import LogisticRegression


class PyLogisticRegression:
    """"
    This class deals with the computation of categorical parameters
    based on a log-linear formulation.

    It is used to produce categorical probabilities as a function of
    a parameter vector (w).
    """

    def __init__(self, feature_matrix: FeatureMatrix,
                 weight_vector: 'np.array',
                 e_vocab_size: int,
                 f_vocab_size: int):
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

    def feature_vector(self, e: int, f: int) -> csr_matrix:
        return self._feature_matrix.get_feature_vector(f, e)

    def probability(self, e: int, f: int) -> float:
        """
        Because we don't have the e and f sentences, we need another formula to calculate theta,
        also because I have not saved the theta's somewhere on forehand
        returns theta_c,d(w) = exp(<w,f(c,d )>)/sum_d'(exp(<w,f(c,d)>))
        :param e: English word
        :param f: French word
        :return: theta associated with e-f pair
        """
        return self.potential(e, f) / self.denominator(e)

    def potential(self, e, f) -> float:
        """
        Calculate the potential (an unnormalised probability) for a given f and e
        :param e: English word
        :param f: French word
        :return: exp(w dot phi)
        """

        potential = self._numerator_cache[e, f]
        if potential < 0:  # potentials can never be negative, thus we haven't computed this one yet
            # compute the potential
            phi = self._feature_matrix.get_feature_vector(f, e)
            potential = np.exp(phi.dot(self._weight_vector)[0])  # phi.dot(w) returns a list/array with one element.
            # store it in the cache
            self._numerator_cache[e, f] = potential
        return potential

    def denominator(self, e) -> float:
        """
        Calculate a denominator Z for a given e, where
            Z = sum_f exp(w * phi(e,f))
        We also cache this computation.

        :param e: English word
        :return: Z
        """
        Z = self._denominator_cache[e]
        if Z < 0:  # this denominator has not yet been computed
            Z = 0.0
            for f in range(self._f_vocab_size):
                Z += self.potential(e, f)
            self._denominator_cache[e] = Z
        return Z


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


class LogLinearParameters(GenerativeComponent):  # Component

    def __init__(self, e_vocab_size, f_vocab_size,
                 weight_vector: np.array,
                 feature_matrix: FeatureMatrix,
                 p=0.0,
                 lbfgs_steps=3,
                 lbfgs_max_attempts=5):
        self._weight_vector = weight_vector  # np.array of w's
        self._feature_matrix = feature_matrix  # dok matrix for each decision (context x features)
        self._e_vocab_size = e_vocab_size
        self._f_vocab_size = f_vocab_size
        self._cpds = sparse.CPDTable(e_vocab_size, f_vocab_size, p)  # expected counts
        self._logistic_regression = LogisticRegression(feature_matrix,
                                                       weight_vector,
                                                       e_vocab_size,
                                                       f_vocab_size)
        self._lbfgs_steps = lbfgs_steps
        self._lbfgs_max_attempts = lbfgs_max_attempts

    def e_vocab_size(self) -> int:
        """
        :return: vocabulary size of English corpus
        """
        return self._e_vocab_size

    def f_vocab_size(self) -> int:
        """
        :return: vocabulary size of English corpus
        """
        return self._f_vocab_size

    def get(self, e_snt: np.array, f_snt: np.array, i: int, j: int) -> float:
        """
        return a parameter (probability) for e[i] (context) and f[j] (decision)
        let's call this parameter theta
        phi(e_i, f_j) is feature function (a vector of feature values)
        w is a parameter vector

        pot(f_j|e_i) = exp(w dot phi(e_i, f_j)) is a potential (a non-negative function of w)
        theta = pot(e_i, f_j) / Z
        Z is the constant that normalizes the potential, this is function of the context
        Z(e_i) = sum for all possible f's in vocab of pot(f'|e_i) (note context is fixed)
        implementation: cache the values of Z (associated with a certain context)

        :param e_snt: English sentence, represented by an np.array of integers
        :param f_snt: French sentence, represented by an np.array of integers
        :param i: The ith position in the English sentence
        :param j: The jth position in the French sentence
        :return: a parameter for p(f[j]|e[i]), represented by float
        """
        return self._logistic_regression.probability(e_snt[i], f_snt[j])

    def plus_equals(self, e_snt: np.array, f_snt: np.array, i: int, j: int, p: float):
        """
        Accumulate fractional counts (p) for events (e_i as context, f_j as decision)
        table[e_i][f_j] += p
        Note, this can be implemented with existing data structures (CPD table)
        Why CPD?: for each context we get a new distribution

        :param e_snt: English sentence, represented by an np.array of integers
        :param f_snt: French sentence, represented by an np.array of integers
        :param i: The ith position in the English sentence
        :param j: The jth position in the French sentence
        :param p: Probability associated with p(f_j|e_i)
        :return:
        """
        return self._cpds.plus_equals(e_snt[i], f_snt[j], p)

    def normalise(self):
        """
        not to confuse with Z(e_i)
        Returns updated weight parameters
        Update weight vector with LBFGS:
        1st step:
                calculate expected feature vector: mu_c = sum over all d's (d')(theta(c,d')*f(c,d'))
        2nd step:
                calculate derivative of expected log likelihood: Delta l(w,e) = sum over all c-d pairs
                (e_c,d) * (f(c,d) - mu_c)
        3rd step:
                climb the gradient with respect to w using regularization term * derivative of expected log likelihood
                (which is the change in w)
        :return:
        """
        w = self.climb()  # this is the REPEAT in the paper
        self._weight_vector = w
        self._logistic_regression = LogisticRegression(self._feature_matrix,
                                                       w,
                                                       self._e_vocab_size,
                                                       self._f_vocab_size)

    def climb(self):

        iteration = 0
        f_calls = 0

        def f(w):
            nonlocal f_calls
            f_calls += 1
            logging.info('[%d] Computing objective and gradient [%d]', iteration, f_calls)
            loglikelihood = ObjectiveAndGradient(self._cpds, self._feature_matrix,
                                 self._e_vocab_size,
                                 self._f_vocab_size)
            # in this function we change the logic from maximisation to minimisation
            objective, gradient = loglikelihood.evaluate(w)
            logging.info('[%d] Objective [%d] %f', iteration, f_calls, objective)
            objective *= -1
            gradient *= -1
            return objective, gradient.toarray()[0]  # weird scipy notation

        def callback(w):
            nonlocal iteration
            iteration += 1
            # can use logging.info(...)
            # to print the current w on the screen
            logging.info('[%d] L-BFGS-B found new weights', iteration)

        # This is our initial vector, namely, the one we used in the E-step
        w0 = self._weight_vector
        logging.info('Optimising log-linear (lexical) parameters for %d steps of L-BFGS-B (max %d evaluations)',
                     self._lbfgs_steps,
                     self._lbfgs_max_attempts)
        result = minimize(f,  # this function returns (for a given w) the negative likelihood and negative gradient
                 w0, # initial vector
                 method='L-BFGS-B',  # variant of GD
                 jac=True,  # yes we are returning the gradient through function f
                 callback=callback,  # this is the function called each time the algorithm finds a new w
                 options={'maxiter': self._lbfgs_steps,  # options of L-BFGS-B
                          'maxfun': self._lbfgs_max_attempts,
                          'ftol': 1e-6,
                          'gtol': 1e-6,
                          'disp': False})
        # you can use logging.info(...) to inspect some stuff like
        # result.fun
        # result.nfev
        # result.nit
        # result.success
        # result.message

        # these are the new weights
        return result.x

    def zeros(self):
        """
        :return: copy of itself, initialized with 0's
        """
        return LogLinearParameters(self.e_vocab_size(), self.f_vocab_size(), self._weight_vector, self._feature_matrix, 0.0)

    def save(self, e_corpus, f_corpus, path):
        # save in '{0}.w'.format(path) the weight vector
        # and in '{0}.cat'.format(path) the output of logistic regression for every e-f pair
        pass

from lola.model import DefaultModel
from lola.component import UniformAlignment, JumpParameters, BrownDistortionParameters

class LogLinearIBM1(DefaultModel):
    """
    An IBM1 is a 0th-order log-linear model with lexical parameters only.
    """

    def __init__(self, lex_parameters: LogLinearParameters):
        super(LogLinearIBM1, self).__init__([lex_parameters, UniformAlignment()])

    def lexical_parameters(self) -> LogLinearParameters:
        return self.component(0)


class VogelLogLinearIBM2(DefaultModel):
    """
    An IBM2 is a 0th-order model with lexical parameters only.
    """

    def __init__(self, lex_parameters: LogLinearParameters, dist_parameters: JumpParameters):
        super(VogelLogLinearIBM2, self).__init__([lex_parameters, dist_parameters])

    def lexical_parameters(self) -> LogLinearParameters:
        return self.component(0)

    def dist_parameters(self) -> JumpParameters:
        return self.component(1)


class BrownLogLinearIBM2(DefaultModel):
    """
    An IBM2 is a 0th-order model with lexical parameters only.
    """

    def __init__(self, lex_parameters: LogLinearParameters, dist_parameters: BrownDistortionParameters):
        super(BrownLogLinearIBM2, self).__init__([lex_parameters, dist_parameters])

    def lexical_parameters(self) -> LogLinearParameters:
        return self.component(0)

    def dist_parameters(self) -> BrownDistortionParameters:
        return self.component(1)
