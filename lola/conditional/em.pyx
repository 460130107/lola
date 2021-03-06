"""
This is a cython implementation of zeroth-order HMM alignment models (e.g. IBM1 and IBM2).


Not using these macros for now:

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

:Authors: - Wilker Aziz
"""

"""
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
"""

import logging
cimport cython
cimport numpy as np
cimport libc.math as c_math
import numpy as np
from lola.corpus cimport Corpus
from lola.conditional.model cimport GenerativeModel
from lola.ptypes cimport uint_t
from lola.ptypes cimport real_t
import lola.ptypes as ptypes


cpdef real_t viterbi_alignments(Corpus e_corpus, Corpus f_corpus, GenerativeModel model, callback):
    """
    This writes Viterbi alignments to an output stream.

    :param e_corpus:
    :param f_corpus:
    :param lex_parameters:
    :param callback: a function to deal with the Viterbi decisions (called for one sentence pair at a time)
        callback(0-based sentence id, alignments, posterior probabilities)
    """
    cdef:
        size_t S = f_corpus.n_sentences()
        size_t s
        uint_t[::1] f_snt, e_snt
        uint_t[::1] alignment = np.zeros(f_corpus.max_len(), dtype=ptypes.uint)
        real_t[::1] posterior = np.zeros(f_corpus.max_len(), dtype=ptypes.real)
        size_t i, j, best_i
        size_t f, e
        real_t p, best_p

    for s in range(S):
        f_snt = f_corpus.sentence(s)
        e_snt = e_corpus.sentence(s)
        m = f_snt.shape[0]
        l = e_snt.shape[0]
        for j in range(m):
            best_i = 0
            best_p = - 1.0
            Z = 0
            for i in range(l):
                p = model.likelihood(e_snt, f_snt, i, j)
                Z += p
                # introduced a deterministic tie-break heuristic that dislikes null-alignments
                if p > best_p:
                    best_p = p
                    best_i = i
            alignment[j] = best_i
            posterior[j] = 0.0 if Z == 0 else best_p / Z
        callback(s, alignment[0:m], posterior[0:m])


cpdef real_t loglikelihood(Corpus e_corpus, Corpus f_corpus, GenerativeModel model):
    """
    Computes the log-likelihood of the data under IBM 1 for given parameters.

    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param model: a 0-order directed graphical model
    :return: log of \prod_{f_1^m,e_0^l} \prod_{j=1}^m \sum_{i=0}^l lex(f_j|e_i)
    """
    cdef:
        real_t loglikelihood = 0.0
        size_t S = f_corpus.n_sentences()
        uint_t[::1] f_snt, e_snt
        size_t s, i, j
        size_t e, f
        float p

    for s in range(S):
        f_snt = f_corpus.sentence(s)
        e_snt = e_corpus.sentence(s)
        for j in range(f_snt.shape[0]):
            p = 0.0
            for i in range(e_snt.shape[0]):
                p += model.likelihood(e_snt, f_snt, i, j)
            loglikelihood += c_math.log(p)

    return loglikelihood


cpdef real_t empirical_cross_entropy(Corpus e_corpus, Corpus f_corpus, GenerativeModel model, real_t log_zero=-99):
    """
    Computes H(p*, p) = - 1/N \sum_{s=1}^N p(f^s|e^s)
        where p is a model distribution
        p* is the unknown true distribution
        and we have N iid samples.

    Note that perplexity can be obtained by computing 2^H(p*,p).

    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param model: a 0-order directed graphical model
    :param log_zero: ln(0) which defaults to -99
    :return: H(p*, p)
    """
    cdef:
        real_t log_p_s
        real_t entropy = 0.0
        size_t S = f_corpus.n_sentences()
        uint_t[::1] f_snt, e_snt
        size_t s, i, j
        size_t e, f
        real_t p

    for s in range(S):
        e_snt = e_corpus.sentence(s)
        f_snt = f_corpus.sentence(s)
        log_p_s = 0.0
        for j in range(f_snt.shape[0]):
            p = 0.0
            for i in range(e_snt.shape[0]):
                p += model.likelihood(e_snt, f_snt, i, j)
            if p == 0:  # entropy is typically computed for test sets as well, where some words can be unknown
                log_p_s += log_zero
            else:
                log_p_s += c_math.log(p)
        entropy += log_p_s

    return - entropy / S


cpdef EM(Corpus e_corpus, Corpus f_corpus, size_t iterations, GenerativeModel model):
    """
    MLE estimates via EM for zeroth-order HMMs.

    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param iterations: a number of iterations
    :param model: a 0-order directed graphical model
    :return: model, entropy log
    """
    cdef size_t iteration

    logging.debug('[%d] Cross entropy', 0)
    cdef real_t H = empirical_cross_entropy(e_corpus, f_corpus, model)
    logging.info('I=%d H=%f', 0, H)
    cdef list entropy_log = [H]

    for iteration in range(iterations):
        logging.debug('[%d] E-step', iteration + 1)
        e_step(e_corpus, f_corpus, model)
        logging.debug('[%d] M-step', iteration + 1)
        model.update()
        logging.debug('[%d] Cross entropy', iteration + 1)
        H = empirical_cross_entropy(e_corpus, f_corpus, model)
        entropy_log.append(H)
        logging.info('I=%d H=%f', iteration + 1, H)

    return entropy_log


@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef e_step(Corpus e_corpus, Corpus f_corpus, GenerativeModel model):
    """
    The E-step gathers expected/potential counts for different types of events updating
    the sufficient statistics.

    IBM1 uses lexical events only.
    IBM2 uses lexical envents and distortion events.

    :param e_corpus:
    :param f_corpus:
    :param model: a zeroth-order HMM
    :param suffstats: a sufficient statistics object compatible with the model
    """

    cdef size_t S = f_corpus.n_sentences()
    cdef uint_t[::1] f_snt, e_snt
    cdef real_t[::1] likelihood_aj = np.zeros(e_corpus.max_len(), dtype=ptypes.real)
    cdef size_t s, i, j
    cdef size_t f, e
    cdef float marginal, p

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

        for j in range(f_snt.shape[0]):
            f = f_snt[j]

            # To compute the probability of each outcome of a_j
            # we start by evaluating the numerator of P(a_j=i|e_0^l,f_1^m) for every possible i
            marginal = 0.0
            for i in range(e_snt.shape[0]):
                # if this was IBM 2, we would also have the contribution of a distortion parameter
                p = model.likelihood(e_snt, f_snt, i, j)
                likelihood_aj[i] = p
                marginal += p

            # Once the (normalised) posterior probability of each outcome has been computed
            #  we can easily gather partial counts
            for i in range(e_snt.shape[0]):
                model.observe(e_snt, f_snt, i, j, likelihood_aj[i] / marginal)

