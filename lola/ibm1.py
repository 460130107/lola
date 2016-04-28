import numpy as np
from lola.corpus import Corpus


def ibm1(f_path, e_path, iterations):
    """
    Estimate IBM1 parameters via EM for a number of iterations starting from uniform parameters
    :param f_path: path to the french corpus text file
    :param e_path: path to the english corpus text file
    :param iterations: amount of iterations for EM
    :return: the optimized lexical parameters
    """

    f_corpus = Corpus(f_path)
    e_corpus = Corpus(e_path, null='<NULL>') # add a NULL word

    lex_param = uniform_lex_param(f_corpus.vocab_size(), e_corpus.vocab_size())

    for i in range(0, iterations):
        lex_counts = e_step(f_corpus, e_corpus, lex_param)
        lex_param = m_step(lex_counts)
        log_likelihood = ibm1_log_likelihood(f_corpus, e_corpus, lex_param)
        print(lex_param)
        print(log_likelihood)
    return lex_param


def e_step(f_corpus, e_corpus, lex_param):
    """
    Estimation step of EM, calculating lexical counts
    :param f_corpus:
    :param e_corpus:
    :param lex_param:
    :return:
    """
    lex_counts = np.zeros((f_corpus.vocab_size(), e_corpus.vocab_size()), dtype=np.float)

    # extract small lexical count matrix for each sentence
    for f_sentence, e_sentence in zip(f_corpus.itersentences(), e_corpus.itersentences()):

        posterior = np.zeros((f_sentence.size, e_sentence.size), dtype=np.float)

        for i, f_word in enumerate(f_sentence):
            for j, e_word in enumerate(e_sentence):
                posterior[i, j] += lex_param[f_word, e_word]
        posterior /= posterior.sum(axis=0)

        # enter the counts in the big lexical parameter matrix
        for i, f_word in enumerate(f_sentence):
            for j, e_word in enumerate(e_sentence):
                lex_counts[f_word, e_word] += posterior[i, j]

    return lex_counts


def m_step(lex_counts):
    """"
    Normalize the lexical counts
    :param lex_counts: lexical counts
    :return: np.array with new lexical parameters
    """
    z = lex_counts.sum(axis=1)
    lex_param = lex_counts / z[:, np.newaxis] # [np.newaxis, :] not necessary?
    return lex_param


def uniform_lex_param(f_vocab_size, e_vocab_size):
    """
    Initialize lexical parameters with a uniform distribution
    :param f_vocab_size: length of french vocabulary
    :param e_vocab_size: length of english vocabulary
    :return:
    """
    init_value = 1.0/f_vocab_size
    lex_param = np.full((f_vocab_size, e_vocab_size), init_value, dtype=np.float)
    return lex_param


def ibm1_log_likelihood(f_corpus, e_corpus, lex_param):
    """
    Calculate log-likelihood of current set of lexical parameters
    :param f_corpus:
    :param e_corpus:
    :param lex_param:
    :return: the log-likelihood for the current lexical parameters
    """
    p = 0.0
    # sum over all sentences
    for f_sentence, e_sentence in zip(f_corpus.itersentences(), e_corpus.itersentences()):
        log_likelihood = 0.0

        # sum over all possible alignments
        for i, f_word in enumerate(f_sentence):
            for j, e_word in enumerate(e_sentence):
                log_likelihood += lex_param[f_word, e_word]
        log_likelihood = np.log(log_likelihood)
        p += log_likelihood
    return p

# french = 'training\hansards.36.2.f'
# english = 'training\hansards.36.2.e'
f = '../training/example.f'
e = '../training/example.e'

ibm1(f, e, 10)
