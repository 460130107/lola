"""
:Authors: - Wilker Aziz
"""
import numpy as np
import sys

from lola.corpus import Corpus
from lola.dist import uniform_lexical


def ibm1_loglikelihood(f_corpus, e_corpus, lex_parameters):
    """
    Computes the log-likelihood of the data under IBM 1 for given parameters.

    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param lex_parameters: a collection of |V_E| categoricals over the French vocabulary V_F
    :return: log of \prod_{f_1^m,e_0^l} \prod_{j=1}^m \sum_{i=0}^l lex(f_j|e_i)
    """
    loglikelihood = 0.0
    for f_snt, e_snt in zip(f_corpus.itersentences(), e_corpus.itersentences()):
        for j, f in enumerate(f_snt):
            p = 0.0
            for i, e in enumerate(e_snt):
                p += lex_parameters[e, f]
            loglikelihood += np.log(p)
    return loglikelihood


def ibm1(f_corpus, e_corpus, iterations):
    """
    Estimate IBM1 parameters via EM for a number of iterations starting from uniform parameters.

    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param iterations: a number of iterations
    :return: MLE lexical parameters
    """

    lex_parameters = uniform_lexical(e_corpus.vocab_size(), f_corpus.vocab_size())

    for iteration in range(iterations):
        lex_counts = e_step(f_corpus, e_corpus, lex_parameters)
        lex_parameters = m_step(lex_counts)

        loglikelihood = ibm1_loglikelihood(f_corpus, e_corpus, lex_parameters)
        print('L{0} {1}'.format(iteration + 1, loglikelihood), file=sys.stderr)

    # TODO: save lexical parameters for inspection

    # Viterbi alignments
    for f_snt, e_snt in zip(f_corpus.itersentences(), e_corpus.itersentences()):
        alignment = np.zeros(f_snt.size, dtype=np.int)
        for j, f in enumerate(f_snt):
            best_i = 0
            best_p = 0
            for i, e in enumerate(e_snt):
                p = lex_parameters[e, f]
                # introduced a deterministic tie-break heuristic that dislikes null-alignments
                if p > best_p or (p == best_p and best_i == 0):
                    best_p = p
                    best_i = i
            alignment[j] = best_i
        # in printing we make the French sentence 1-based by convention
        # we keep the English sentence 0-based because of the NULL token
        print(' '.join('{0}-{1}'.format(j + 1, i) for j, i in enumerate(alignment)))

    return lex_parameters


def e_step(f_corpus, e_corpus, lex_parameters):
    """
    The E-step gathers expected/potential counts for different types of events.
    IBM1 uses lexical events only.
    IBM2 uses lexical envents and distortion events.

    :param f_corpus:
    :param e_corpus:
    :param lex_parameters:
    :return:
    """

    lex_counts = np.zeros(lex_parameters.shape, dtype=np.float)

    for f_snt, e_snt in zip(f_corpus.itersentences(), e_corpus.itersentences()):

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
        # Thus a cell p_ij[i,j] is associated with P(a_j=i|e_0^l,f_1^m)
        posterior = np.zeros((e_snt.size, f_snt.size))

        # To compute each cell we start by evaluating the numerator of P(a_j=i|e_0^l,f_1^m) for every possible (i,j)
        for j, f in enumerate(f_snt):
            for i, e in enumerate(e_snt):
                # if this was IBM 2, we would also have the contribution of a distortion parameter
                posterior[i, j] += lex_parameters[e, f]
        # Then we renormalise each column independently by the sum along that column
        posterior /= posterior.sum(0)

        # Once the (normalised) posterior probability of each candidate alignment link has been computed
        #  we can easily gather partial counts
        for j, f in enumerate(f_snt):
            for i, e in enumerate(e_snt):
                lex_counts[e, f] += posterior[i, j]
                # if this was IBM2, we would also accumulate dist_counts[i, j] += posterior

    return lex_counts


def m_step(lex_counts):
    """
    The M-step simply renormalise potential counts.

    :param lex_counts: potential counts of lexical events.
    :return: locally optimum lexical parameters
    """
    # we compute normalisation constants for each English word by summing the cells along the corresponding row
    Z = lex_counts.sum(1)
    # then we divide each row by the corresponding normalisation constant
    return lex_counts / Z[:,np.newaxis]


def main(f_path, e_path):
    f_corpus = Corpus(f_path)
    e_corpus = Corpus(e_path, null='<NULL>')
    ibm1(f_corpus, e_corpus, 10)


if __name__ == '__main__':
    main('training/example.f', 'training/example.e')