
from lola.event cimport Event
from lola.component cimport GenerativeComponent
from lola.fmatrix cimport SparseFeatureMatrix, DenseFeatureMatrix
from lola.fmatrix cimport make_cpds, make_cpds2
from lola.event cimport EventSpace
from lola.corpus cimport Corpus
cimport numpy as np
cimport cython

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import scipy as sp
import numpy as np
import logging


cdef class ObjectiveAndGradient:

    cdef:
        DenseFeatureMatrix _dense_matrix
        SparseFeatureMatrix _sparse_matrix
        EventSpace _event_space
        object _sparse_expected_counts

    def __init__(self, expected_counts,
                 DenseFeatureMatrix dense_matrix,
                 SparseFeatureMatrix sparse_matrix,
                 EventSpace event_space):
        """

        :param expected_counts: this is the result of the E-step (that is, the
            part that remains unchanged during the M-step)
        """
        self._dense_matrix = dense_matrix
        self._sparse_matrix = sparse_matrix
        self._event_space = event_space
        self._sparse_expected_counts = csr_matrix(expected_counts)

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
        wd = weight_vector[:self._dense_matrix.dimensionality()]
        ws = weight_vector[self._dense_matrix.dimensionality():]
        cdef np.float_t[:, ::1] cpds = make_cpds2(wd, ws,
                                                  self._dense_matrix, self._sparse_matrix,
                                                  self._event_space.n_contexts(),
                                                  self._event_space.n_decisions())
        cdef float mass, objective

        gradient = sp.matrix(np.zeros(weight_vector.shape[0]))
        objective = 0.0

        for ctxt in range(self._event_space.n_contexts()):

            # First we compute the expected feature vector for a given context



            #mu_fvec = self._sparse_matrix.feature_matrix(ctxt).T * cpds[ctxt]

            # much faster version
            mass = self._sparse_expected_counts[ctxt].sum()

            update = np.zeros(weight_vector.shape[0])
            if self._dense_matrix.dimensionality():
                d = self._sparse_expected_counts[ctxt].dot(self._dense_matrix.feature_matrix(ctxt))
                d_mu = self._dense_matrix.expected_fvector(ctxt, cpds[ctxt])
                d_update = (d - d_mu * mass)[0]
                update[:self._dense_matrix.dimensionality()] = d_update

            if self._sparse_matrix.dimensionality():
                s = self._sparse_expected_counts[ctxt].dot(self._sparse_matrix.feature_matrix(ctxt))
                s_mu = self._sparse_matrix.expected_fvector(ctxt, cpds[ctxt])
                s_update = (s - s_mu * mass).A[0]
                update[self._dense_matrix.dimensionality():] = s_update

            #update = self._sparse_expected_counts[ctxt].dot(self._sparse_matrix.feature_matrix(ctxt)) - mu_fvec * self._sparse_expected_counts[ctxt].sum()

            gradient += update

            # This is the expected log-likelihood
            objective += self._sparse_expected_counts[ctxt].dot(np.log(cpds[ctxt]))[0]  # matrix (1,) -> scalar

        # if regulariser_strength != 0.0:
        #   objective -= regulariser_strength * squared(l2_norm(w))
        #   gradient -= 2 * regulariser_strength * w

        return objective, gradient.A[0]  # matrix to standard np.array



cdef float desc_sort_by_weight(tuple pair):
    return -pair[1]

cdef class LogLinearComponent(GenerativeComponent):  # Component

    cdef:
        np.float_t[::1] _wd
        np.float_t[::1] _ws
        DenseFeatureMatrix _dense_matrix
        SparseFeatureMatrix _sparse_matrix
        EventSpace _event_space
        np.float_t[:,::1] _cpds
        object _sparse_counts
        int _lbfgs_steps
        int _lbfgs_max_attempts


    def __init__(self, np.float_t[::1] wd, np.float_t[::1] ws,
                 DenseFeatureMatrix dense_matrix,
                 SparseFeatureMatrix sparse_matrix,
                 EventSpace event_space,
                 int lbfgs_steps=3,
                 int lbfgs_max_attempts=5,
                 name='LogLinearComponent'):
        super(LogLinearComponent, self).__init__(name)
        #logging.info('w_d=%s', np.array(wd))
        #logging.info('w_s=%s', np.array(ws))
        self._wd = wd
        self._ws = ws
        self._dense_matrix = dense_matrix
        self._sparse_matrix = sparse_matrix
        self._event_space = event_space

        self._cpds = make_cpds2(wd,
                                 ws,
                                 self._dense_matrix,
                                 self._sparse_matrix,
                                 self._event_space.n_contexts(),
                                 self._event_space.n_decisions())
        self._sparse_counts = lil_matrix(event_space.shape())  # expected counts
        self._lbfgs_steps = lbfgs_steps
        self._lbfgs_max_attempts = lbfgs_max_attempts

    cpdef EventSpace event_space(self):
        return self._event_space

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
        cdef Event event = self._event_space.get(e_snt, f_snt, i, j)
        return self._cpds[event.context.id, event.decision.id]

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
        cdef Event event = self._event_space.get(e_snt, f_snt, i, j)
        self._sparse_counts[event.context.id, event.decision.id] += p
        return self._sparse_counts[event.context.id, event.decision.id]

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
        self._wd, self._ws = self.climb()  # this is the REPEAT in the paper
        #logging.info('Optimised w_d=%s', np.array(self._wd))
        #logging.info('Optimised w_s=%s', np.array(self._ws))
        self._cpds = make_cpds2(self._wd, self._ws,
                                self._dense_matrix, self._sparse_matrix,
                                self._event_space.n_contexts(),
                                self._event_space.n_decisions())

    def climb(self):

        iteration = 0
        f_calls = 0

        def f(w):
            nonlocal f_calls
            f_calls += 1
            logging.info('[%d] Computing objective and gradient [%d]', iteration, f_calls)
            #logging.info('[%d] input weights [%d]: %s', iteration, f_calls, w)
            loglikelihood = ObjectiveAndGradient(self._sparse_counts,
                                                 self._dense_matrix,
                                                 self._sparse_matrix,
                                                 self._event_space)
            # in this function we change the logic from maximisation to minimisation
            objective, gradient = loglikelihood.evaluate(w)
            logging.info('[%d] Objective [%d] %f', iteration, f_calls, objective)
            objective *= -1
            gradient *= -1
            return objective, gradient

        def callback(w):
            nonlocal iteration
            iteration += 1
            # can use logging.info(...)
            # to print the current w on the screen
            logging.info('[%d] L-BFGS-B found new weights', iteration)
            #logging.debug('[%d] weigthts: %s', iteration, w)

        # This is our initial vector, namely, the one we used in the E-step
        w0 = np.concatenate((self._wd, self._ws))
        logging.info('Optimising log-linear parameters (%s) for %d steps of L-BFGS-B (max %d evaluations)',
                     self.name(),
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
        return result.x[:self._dense_matrix.dimensionality()], result.x[self._dense_matrix.dimensionality():]

    cpdef GenerativeComponent zeros(self):
        """
        :return: copy of itself, initialized with 0's
        """
        return LogLinearComponent(self._wd, self._ws,
                                  self._dense_matrix, self._sparse_matrix,
                                  self._event_space, self._lbfgs_steps, self._lbfgs_max_attempts,
                                  self._name)

    cpdef save(self, Corpus e_corpus, Corpus f_corpus, str path):
        # save in '{0}.w'.format(path) the weight vector
        # and in '{0}.cat'.format(path) the output of logistic regression for every e-f pair
        with open('{0}.{1}.sparse'.format(path, self.name()), 'w') as fo:
            for fid, w in sorted(enumerate(self._ws), key=desc_sort_by_weight):
                print('{0}\t{1}\t{2}'.format(fid, self._sparse_matrix.raw_feature_value(fid), w), file=fo)
        with open('{0}.{1}.dense'.format(path, self.name()), 'w') as fo:
            for fid, w in sorted(enumerate(self._wd), key=desc_sort_by_weight):
                print('{0}\t{1}\t{2}'.format(fid, self._dense_matrix.descriptor(fid), w), file=fo)

