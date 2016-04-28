"""
This is a cython implementation of IBM model 1.
It runs a lot faster than its python equivalent (wibm1.py).

:Authors: - Wilker Aziz
"""

import numpy as np
cimport numpy as np
cimport cython
import sys
import logging

from lola.dist import uniform_lexical
from lola.corpus cimport Corpus


cpdef viterbi1(Corpus f_corpus, Corpus e_corpus, np.float_t[:,::1] lex_parameters, ostream=sys.stdout):
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
        np.int_t[:,::1] alignment
        size_t i, j, best_i
        int f, e
        float p, best_p

    for s in range(S):
        f_snt = f_corpus.sentence(s)
        e_snt = e_corpus.sentence(s)
        alignment = np.zeros(f_snt.size, dtype=np.int)
        for j in range(len(f_snt)):
            f = f_snt[j]
            best_i = 0
            best_p = 0
            for i in range(len(e_snt)):
                e = e_snt[i]
                p = lex_parameters[e, f]
                # introduced a deterministic tie-break heuristic that dislikes null-alignments
                if p > best_p or (p == best_p and best_i == 0):
                    best_p = p
                    best_i = i
            alignment[j] = best_i
        # in printing we make the French sentence 1-based by convention
        # we keep the English sentence 0-based because of the NULL token
        print(' '.join(['{0}-{1}'.format(j + 1, i) for j, i in enumerate(alignment)]), file=ostream)


@cython.linetrace(False)
cdef float ibm1_loglikelihood(Corpus f_corpus, Corpus e_corpus, np.float_t[:,::1] lex_parameters):
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
                p += lex_parameters[e, f]
            loglikelihood += np.log(p)

    return loglikelihood


@cython.linetrace(False)
cpdef np.float_t[:,::1] ibm1(Corpus f_corpus, Corpus e_corpus, int iterations, bint viterbi=True):
    """
    Estimate IBM1 parameters via EM for a number of iterations starting from uniform parameters.

    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param iterations: a number of iterations
    :param viterbi: whether or not to print Viterbi alignments after optimisation
    :return: MLE lexical parameters
    """

    cdef np.float_t[:,::1] lex_parameters = uniform_lexical(e_corpus.vocab_size(), f_corpus.vocab_size())
    cdef np.float_t[:,::1] lex_counts
    cdef size_t iteration

    loglikelihood = ibm1_loglikelihood(f_corpus, e_corpus, lex_parameters)
    print('L{0} {1}'.format(0, loglikelihood), file=sys.stderr)

    for iteration in range(iterations):
        lex_counts = e_step(f_corpus, e_corpus, lex_parameters)
        lex_parameters = m_step(lex_counts)
        loglikelihood = ibm1_loglikelihood(f_corpus, e_corpus, lex_parameters)
        print('L{0} {1}'.format(iteration + 1, loglikelihood), file=sys.stderr)

    # TODO: save lexical parameters for inspection

    if viterbi:  # Viterbi alignments
        viterbi1(f_corpus, e_corpus, lex_parameters)

    return lex_parameters


@cython.linetrace(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float_t[:,::1] e_step(Corpus f_corpus, Corpus e_corpus, np.float_t[:,::1] lex_parameters):
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
    cdef np.float_t[:,::1] lex_counts = np.zeros(np.shape(lex_parameters), dtype=np.float)
    cdef np.int_t[::1] f_snt, e_snt
    cdef np.float_t[:,::1] posterior
    cdef size_t s, i, j
    cdef int f, e

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
        # I am choosing to represent it by a table where rows are indexed by English positions
        #  and columns are indexed by French positions
        # Thus a cell posterior[i,j] is associated with P(a_j=i|e_0^l,f_1^m)
        posterior = np.zeros((np.size(e_snt), np.size(f_snt)), dtype=np.float)

        # To compute each cell we start by evaluating the numerator of P(a_j=i|e_0^l,f_1^m) for every possible (i,j)
        for j in range(len(f_snt)):
            f = f_snt[j]
            for i in range(len(e_snt)):
                e = e_snt[i]
                # if this was IBM 2, we would also have the contribution of a distortion parameter
                posterior[i, j] += lex_parameters[e, f]
        # Then we renormalise each column independently by the sum along that column
        posterior /= np.sum(posterior, 0)

        # Once the (normalised) posterior probability of each candidate alignment link has been computed
        #  we can easily gather partial counts
        for j in range(len(f_snt)):
            f = f_snt[j]
            for i in range(len(e_snt)):
                e = e_snt[i]
                lex_counts[e, f] += posterior[i, j]
                # if this was IBM2, we would also accumulate dist_counts[i, j] += posterior

        if (s + 1) % 1000 == 0:
            logging.info('E-step %d/%d sentences', s, S)

    return lex_counts


@cython.linetrace(False)
cdef np.float_t[:,::1] m_step(np.float_t[:,::1] lex_counts):
    """
    The M-step simply renormalise potential counts.

    :param lex_counts: potential counts of lexical events.
    :return: locally optimum lexical parameters
    """
    # we compute normalisation constants for each English word by summing the cells along the corresponding row
    Z = np.sum(lex_counts, 1)
    # then we divide each row by the corresponding normalisation constant
    # the strange syntax is a requirement of numpy, see structural indexing tools in
    #  http://docs.scipy.org/doc/numpy-1.10.1/user/basics.indexing.html
    return lex_counts / Z[:, np.newaxis]
