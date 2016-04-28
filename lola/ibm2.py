import numpy as np
from lola.corpus import Corpus
import lola.dist as distortion
import lola.ibm1 as ibm1


def ibm2(f_path, e_path, iterations_ibm1, iterations_ibm2):
    """
    Estimate IBM2 parameters via EM for a number of iterations starting from uniform parameters for distortion and
    parameter estimation for lexical parameters using IBM1
    :param f_path: path to the french corpus text file
    :param e_path: path to the english corpus text file
    :param iterations_ibm1: amount of iterations for parameter estimation of the lexical parameters in model 2
    :param iterations_ibm2: amount of iterations for EM in model 2
    :return: the optimized lexical parameters
    """

    f_corpus = Corpus(f_path)
    e_corpus = Corpus(e_path, null='<NULL>') # add a NULL word

    # estimate lexical parameters using model 1
    lex_param = ibm1.ibm1(f, e, iterations_ibm1)
    dist_param = distortion.uniform_lexical(f_corpus.vocab_size(), e_corpus.vocab_size())
    for i in range(0, iterations_ibm2):
        lex_counts, dist_counts = e_step(f_corpus, e_corpus, lex_param, dist_param)
        lex_param, dist_param = m_step(lex_counts, dist_counts)
        log_likelihood = ibm2_log_likelihood(f_corpus, e_corpus, lex_param, dist_param)
        print(dist_param)
        print(log_likelihood)
    return lex_param, dist_param


def e_step(f_corpus, e_corpus, lex_param, dist_param):
    """
    Estimation step of EM, calculating lexical counts
    :param f_corpus:
    :param e_corpus:
    :param lex_param:
    :param dist_param:
    :return: np.array with lexical and np.array with distortion counts
    """
    lex_counts = np.zeros((f_corpus.vocab_size(), e_corpus.vocab_size()), dtype=np.float)
    dist_counts = lex_counts

    # extract small lexical count matrix for each sentence
    for f_sentence, e_sentence in zip(f_corpus.itersentences(), e_corpus.itersentences()):

        posterior = np.zeros([len(f_sentence), len(e_sentence)], dtype=np.float)

        for i, f_word in enumerate(f_sentence):
            for j, e_word in enumerate(e_sentence):
                posterior[i, j] += (lex_param[f_word, e_word] * dist_param[f_word, e_word])
        posterior /= posterior.sum(axis=0)

        # enter the counts in the big lexical parameter matrix
        for i, f_word in enumerate(f_sentence):
            for j, e_word in enumerate(e_sentence):
                lex_counts[f_word, e_word] += posterior[i, j]
                dist_counts[f_word, e_word] += posterior[i, j]
    return lex_counts, dist_counts


def m_step(lex_counts, dist_counts):
    """"
    Normalize the lexical counts
    :param lex_counts: lexical counts
    :param dist_counts: distortion counts
    :return: np.array with new lexical parameters
    """
    z_lex = lex_counts.sum(axis=1)
    z_dist = dist_counts.sum(axis=1)
    lex_param = lex_counts / z_lex[:, np.newaxis]
    dist_param = dist_counts / z_dist[:, np.newaxis]
    return lex_param, dist_param


def ibm2_log_likelihood(f_corpus, e_corpus, lex_param, dist_param):
    """
    Calculate log-likelihood of current set of lexical parameters and distortion parameters
    :param f_corpus:
    :param e_corpus:
    :param lex_param:
    :param dist_param:
    :return: the log-likelihood for the current lexical parameters and distortion parameters
    """
    p = 0.0
    # sum over all sentences
    for f_sentence, e_sentence in zip(f_corpus.itersentences(), e_corpus.itersentences()):
        log_likelihood = 0.0

        for i, f_word in enumerate(f_sentence):
            # sum over all possible alignments
            for j, e_word in enumerate(e_sentence):
                log_likelihood += lex_param[f_word, e_word] * dist_param[f_word, e_word]
        log_likelihood = np.log(log_likelihood)
        p += log_likelihood
    return p

# french = 'training\hansards.36.2.f'
# english = 'training\hansards.36.2.e'
f = '../training/example.f'
e = '../training/example.e'

ibm2(f, e, 10, 10)
