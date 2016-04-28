"""
This is a cython implementation of IBM model 1 using sparse categoricals.
It runs a lot faster than its python equivalent (wibm1.py).

:Authors: - Wilker Aziz
"""

import numpy as np
cimport numpy as np
cimport cython
import sys
import logging

from lola.corpus cimport Corpus
from lola.sparse cimport SparseCategorical

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


cdef class LexicalParameters:

    cdef list _cpds

    def __init__(self, int e_vocab_size, int f_vocab_size, float p=0.0):
        self._cpds = [SparseCategorical(f_vocab_size, p) for _ in range(e_vocab_size)]

    cpdef float get(self, int e, int f):
        return self._cpds[e].get(f)
    
    cpdef float acc(self, int e, int f, float value):
        return self._cpds[e].acc(f, value)

    cpdef SparseCategorical row(self, int e):
        return self._cpds[e]

    cpdef normalise(self):
        cdef SparseCategorical cpd
        for cpd in self._cpds:
            cpd.normalise()


cpdef viterbi1(Corpus f_corpus, Corpus e_corpus, LexicalParameters lex_parameters, ostream=sys.stdout):
    """
    This writes IBM 1 Viterbi alignments to an output stream.

    :param f_corpus:
    :param e_corpus:
    :param lex_parameters:
    :param ostream: output stream
    """
    cdef:
        size_t S = f_corpus.n_sentences()
        size_t s
        np.int_t[::1] f_snt, e_snt
        np.int_t[::1] alignment
        size_t i, j, best_i
        int f, e
        float p, best_p

    for s in range(S):
        f_snt = f_corpus.sentence(s)
        e_snt = e_corpus.sentence(s)
        alignment = np.zeros(len(f_snt), dtype=np.int)
        for j in range(len(f_snt)):
            f = f_snt[j]
            best_i = 0
            best_p = 0
            for i in range(len(e_snt)):
                e = e_snt[i]
                p = lex_parameters.get(e, f)
                # introduced a deterministic tie-break heuristic that dislikes null-alignments
                if p > best_p or (p == best_p and best_i == 0):
                    best_p = p
                    best_i = i
            alignment[j] = best_i
        # in printing we make the French sentence 1-based by convention
        # we keep the English sentence 0-based because of the NULL token
        print(' '.join(['{0}-{1}'.format(j + 1, i) for j, i in enumerate(alignment)]), file=ostream)


@cython.linetrace(True)
cdef float ibm1_loglikelihood(Corpus f_corpus, Corpus e_corpus, LexicalParameters lex_parameters):
    """
    Computes the log-likelihood of the data under IBM 1 for given parameters.

    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param lex_parameters: a collection of |V_E| categoricals over the French vocabulary V_F
    :return: log of \prod_{f_1^m,e_0^l} \prod_{j=1}^m \sum_{i=0}^l lex(f_j|e_i)
    """
    cdef:
        float loglikelihood = 0.0
        size_t S = f_corpus.n_sentences()
        np.int_t[::1] f_snt, e_snt
        size_t s, i, j
        int e, f
        float p

    for s in range(S):
        f_snt = f_corpus.sentence(s)
        e_snt = e_corpus.sentence(s)
        for j in range(len(f_snt)):
            f = f_snt[j]
            p = 0.0
            for i in range(len(e_snt)):
                e = e_snt[i]
                p += lex_parameters.get(e, f)
            loglikelihood += np.log(p)

    return loglikelihood


@cython.linetrace(True)
cpdef LexicalParameters ibm1(Corpus f_corpus, Corpus e_corpus, int iterations, bint viterbi=True):
    """
    Estimate IBM1 parameters via EM for a number of iterations starting from uniform parameters.

    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param iterations: a number of iterations
    :param viterbi: whether or not to print Viterbi alignments after optimisation
    :return: MLE lexical parameters
    """
    # create |V_E| categoricals each over V_F and initialise them uniformly
    cdef LexicalParameters lex_parameters = LexicalParameters(e_corpus.vocab_size(), f_corpus.vocab_size(), p=1.0/f_corpus.vocab_size())
    cdef size_t iteration

    loglikelihood = ibm1_loglikelihood(f_corpus, e_corpus, lex_parameters)
    logging.info('L%d %f', 0, loglikelihood)

    for iteration in range(iterations):
        lex_parameters = EM_iteration(f_corpus, e_corpus, lex_parameters)
        loglikelihood = ibm1_loglikelihood(f_corpus, e_corpus, lex_parameters)
        logging.info('L%d %f', iteration + 1, loglikelihood)

    # TODO: save lexical parameters for inspection

    if viterbi:  # Viterbi alignments
        viterbi1(f_corpus, e_corpus, lex_parameters)

    return lex_parameters


@cython.linetrace(True)
cdef LexicalParameters EM_iteration(Corpus f_corpus, Corpus e_corpus, LexicalParameters lex_parameters):
    """
    The E-step gathers expected/potential counts for different types of events.
    IBM1 uses lexical events only.
    IBM2 uses lexical envents and distortion events.

    :param f_corpus:
    :param e_corpus:
    :param lex_parameters:
    :return:
    """

    cdef size_t S = f_corpus.n_sentences()
    cdef LexicalParameters lex_counts = LexicalParameters(e_corpus.vocab_size(), f_corpus.vocab_size(), 0.0)
    cdef np.int_t[::1] f_snt, e_snt
    cdef np.float_t[::1] posterior_aj
    cdef size_t s, i, j
    cdef int f, e

    # E-step
    for s in range(S):
        f_snt = f_corpus.sentence(s)
        e_snt = e_corpus.sentence(s)

        # Remember, this is our model
        #   P(f_1^m a_1^m, m|e_0^l) \propto \prod_{j=1}^m P(a_j|m,l) \times P(f_j|e_{a_j})
        # Under IBM models 1 and 2, alignments are independent of one another,
        # thus the posterior of alignment links factorise
        #   P(a_1^m|e_0^l, f_1^m) \propto \prod_{j=1}^m P(a_j|e_0^l, f_1^m)
        # where
        #   P(a_j=i|e_0^l, f_1^m) \propto P(a_j=i|m,l) \times P(f_j|e_i)
        # If this is IBM model 1,
        #   then the first term is a constant (uniform probability over alignments)
        #    which can therefore be ignored (due to proportionality)
        #   and the second term is a lexical Categorical distribution
        # IBM 1:
        #   P(a_j=i|e_0^l,f_1^m) \propto lex(f_j|e_i)

        # Observe that the posterior for a candidate alignment link is a distribution over assignments of a_j=i
        # That is, a French *position* j being aligned to an English *position* i.
        # For a given French position j, I am choosing to represent it by a vector indexed by English positions
        # Thus a cell posterior_aj[i] is associated with P(a_j=i|e_0^l,f_1^m)

        for j in range(len(f_snt)):
            f = f_snt[j]
            posterior_aj = np.zeros(len(e_snt))

            # To compute the probability of each outcome of a_j
            # we start by evaluating the numerator of P(a_j=i|e_0^l,f_1^m) for every possible i
            for i in range(len(e_snt)):
                e = e_snt[i]
                # if this was IBM 2, we would also have the contribution of a distortion parameter
                posterior_aj[i] = lex_parameters.get(e, f)
            # Then we normalise it making a proper cpd
            posterior_aj /= np.sum(posterior_aj)

            # Once the (normalised) posterior probability of each outcome has been computed
            #  we can easily gather partial counts
            for i in range(len(e_snt)):
                e = e_snt[i]
                lex_counts.acc(e, f, posterior_aj[i])
                # if this was IBM2, we would also accumulate dist_counts.acc(i, j, posterior_aj[i])

        #if (s + 1) % 10000 == 0:
        #    logging.debug('E-step %d/%d sentences', s + 1, S)

    # M-step
    lex_counts.normalise()
    return lex_counts

