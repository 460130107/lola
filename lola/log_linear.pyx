import numpy as np
cimport numpy as np
cimport cython
from lola.logreg cimport LogisticRegression
from lola.corpus cimport Corpus
from lola.component cimport GenerativeComponent
from lola.sparse cimport CPDTable

from lola.frepr cimport LexicalFeatureMatrix
from scipy.optimize import minimize
import logging
from lola.gradient import ObjectiveAndGradient

from scipy.sparse import lil_matrix


cdef class LogLinearParameters(GenerativeComponent):  # Component

    cdef:
        np.float_t[::1] _weight_vector
        LexicalFeatureMatrix _feature_matrix
        size_t _e_vocab_size
        size_t _f_vocab_size
        int _lbfgs_steps
        int _lbfgs_max_attempts
        LogisticRegression _logistic_regression
        CPDTable _cpds
        object _sparse_counts

    def __init__(self, size_t e_vocab_size,
                 size_t f_vocab_size,
                 np.float_t[::1] weight_vector,
                 LexicalFeatureMatrix feature_matrix,
                 float p=0.0,
                 int lbfgs_steps=3,
                 int lbfgs_max_attempts=5,
                 name='llLexical'):
        super(LogLinearParameters, self).__init__(name)
        self._weight_vector = weight_vector  # np.array of w's
        self._feature_matrix = feature_matrix  # dok matrix for each decision (context x features)
        self._e_vocab_size = e_vocab_size
        self._f_vocab_size = f_vocab_size
        self._cpds = CPDTable(e_vocab_size, f_vocab_size, p)  # expected counts
        self._logistic_regression = LogisticRegression(feature_matrix,
                                                       weight_vector,
                                                       e_vocab_size,
                                                       f_vocab_size)
        self._lbfgs_steps = lbfgs_steps
        self._lbfgs_max_attempts = lbfgs_max_attempts
        self._sparse_counts = lil_matrix((e_vocab_size, f_vocab_size))

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cpdef float get(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
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

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cpdef float plus_equals(self,  np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
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
        self._sparse_counts[e_snt[i], f_snt[j]] += p
        return self._cpds.plus_equals(e_snt[i], f_snt[j], p)

    cpdef normalise(self):
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
            loglikelihood = ObjectiveAndGradient(self._cpds, self._sparse_counts, self._feature_matrix,
                                 self._e_vocab_size,
                                 self._f_vocab_size)
            # in this function we change the logic from maximisation to minimisation
            objective, gradient = loglikelihood.evaluate(w)
            logging.info('[%d] Objective [%d] %f', iteration, f_calls, objective)
            objective *= -1
            gradient *= -1
            return objective, gradient.A[0]  # matrix to standard np.array

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

    cpdef GenerativeComponent zeros(self):
        """
        :return: copy of itself, initialized with 0's
        """
        return LogLinearParameters(self._e_vocab_size, self._f_vocab_size,
                                   self._weight_vector, self._feature_matrix, 0.0)

    cpdef save(self, Corpus e_corpus, Corpus f_corpus, str path):
        # save in '{0}.w'.format(path) the weight vector
        # and in '{0}.cat'.format(path) the output of logistic regression for every e-f pair
        with open('{0}.{1}'.format(path, self.name()), 'w') as fo:
            for fid, w in enumerate(self._weight_vector):
                print('{0}\t{1}\t{2}'.format(fid, self._feature_matrix.raw_feature_value(fid), w), file=fo)
