from lola.component import GenerativeComponent
import lola.sparse as sparse
import numpy as np
from numpy import linalg as la
import collections



class LogLinearParameters(GenerativeComponent): #Component

    def __init__(self, e_corpus, f_corpus, weight_vector, feature_matrix, p):
        self._weight_vector = weight_vector # np.array of w's
        self._feature_matrix = feature_matrix #dok matrix for each decision (context x features)
        self._e_corpus = e_corpus
        self._f_corpus = f_corpus
        self._cache = collections.defaultdict() # dictionary with Z-values
        self._cpds = sparse.CPDTable(e_corpus.vocab_size(), f_corpus.vocab_size(), p) # expected counts


    def e_vocab_size(self):
        """
        :return: vocabulary size of English corpus
        """
        return self._e_corpus.vocab_size()

    def f_vocab_size(self):
        """
        :return: vocabulary size of English corpus
        """
        return self._f_corpus.vocab_size()

    def get(self, e_snt, f_snt, i: int, j: int):
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
        potential = self.calculate_potential(f_snt[j], e_snt[i])
        Z = self.return_Z(e_snt[i])
        return potential / Z

    def calculate_theta(self, e: int, f: int):
        """
        Because we don't have the e and f sentences, we need another formula to calculate theta,
        also because I have not saved the theta's somewhere on forehand
        returns theta_c,d(w) = exp(<w,f(c,d )>)/sum_d'(exp(<w,f(c,d)>))
        :param e: English word
        :param f: French word
        :return: theta associated with e-f pair
        """
        potential = self.calculate_potential(f, e)
        Z = self.return_Z(e)
        return potential / Z


    def calculate_potential(self, f, e):
        """
        Calculate the potential for a given f and e
        :param f: French word
        :param e: English word
        :return: exp(w dot phi)
        """
        phi = self._feature_matrix.get_feature_vector(f, e)
        return np.exp(np.dot(self._weight_vector, phi))

    def return_Z(self, e):
        """
        Gets Z if cached, else calculates it
        :param e: English word
        :return: return Z
        """
        if e in self._cache:
            return self._cache[e]
        else:
            Z = self.calculate_Z(e)
            return Z

    def calculate_Z(self, e):
        """
        Calculate Z for a given e
        :param e: English word
        :return: Z
        """
        Z = 0.0
        for f in range(self._f_corpus.vocab_size()):
            Z += self.calculate_potential(f, e)
        self._cache[e] = Z
        return Z

    def empty_cache(self):
        """
        Empties cache
        :return:
        """
        self._cache = collections.defaultdict()

    def plus_equals(self, e_snt, f_snt, i, j, p):
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
        expected_feature_vectors = self.calculate_expected_feature_vectors()
        expected_log_likelihood, gradient = self.likelihood_gradient(expected_feature_vectors)
        return expected_log_likelihood, gradient

    def calculate_expected_feature_vectors(self):
        """
        calculates expected feature vectors: mu_c = sum_d'(theta(c,d')*f(c,d')) for each context e
        :return: dictionary of expected feature vectors for each context
        """
        expected_feature_vectors = collections.defaultdict()
        for e in range(self.e_vocab_size()):
            mu = 0
            # loop over all decisions f and sum (theta * feature_vector)
            for f in range(self.f_vocab_size()):
                theta = self._cpds.get(e, f)
                feature_vector = self._feature_matrix.get_feature_vector(f, e)
                mu += theta * feature_vector
            # add vector to dictionary, which can be found by its context
            expected_feature_vectors[e] = mu
        return expected_feature_vectors

    def likelihood_gradient(self, expected_feature_vectors):
        """
        computes both the expected log-likelihood as well as the derivative of it (gradient)
        :param expected_feature_vectors:
        :return: expected log-likelihood and gradient
        """
        gradient = 0
        expected_log_likelihood = 0

        for e in self.e_vocab_size():
            expected_feature_vector = expected_feature_vectors[e]

            for f in self.f_vocab_size():
                expected_counts = self._cpds.get(e, f) # expected counts from e-step
                theta = self.calculate_theta(e, f) # theta_c,d(w) for each w (thus theta is a vector w's)
                feature_vector = self._feature_matrix.get_feature_vector(f, e)
                gradient += expected_counts * (feature_vector - expected_feature_vector)
                expected_log_likelihood += expected_counts * np.log(theta)
        return expected_log_likelihood, gradient

    def zeros(self):
        """
        :return: copy of itself, initialized with 0's
        """
        return LogLinearParameters(self.e_vocab_size(), self.f_vocab_size(), 0.0)





