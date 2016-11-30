"""
:Authors: - Wilker Aziz
"""
import numpy as np
import logging
from lola.corpus import Corpus
import lola.joint.cat as cat
import lola.ptypes as ptypes
import sys
from functools import partial


class ConditionalModel:
    def __init__(self, PL: cat.LengthDistribution,
                 PM: cat.LengthDistribution,
                 PAj: cat.AlignmentDistribution,
                 PFj: cat.SourceDistribution):
        self.PL = PL
        self.PM = PM
        self.PAj = PAj
        self.PFj = PFj
        self.components = tuple([PL, PM, PAj, PFj])

    def update(self):
        for comp in self.components:
            comp.update()


def marginal_likelihood(e_corpus: Corpus, f_corpus: Corpus, model: ConditionalModel):

    PL, PM, PAj, PFj = model.components
    ll = 0.0
    for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
        # observations
        l = e_snt.shape[0]
        m = f_snt.shape[0]
        log_pl = np.log(PL.generate(l))
        log_pm = np.log(PM.generate(m))

        # P(f|e) = \prod_j P(f_j|e)
        #          = \prod_j \sum_i P(f_j,a_j=i|e)
        log_pf_e = 0.0
        for j, f in enumerate(f_snt):
            # P(f_j|e) = \sum_i P(f_j,a_j=i|e)
            pfj_e = 0.0  # contribution of this French word
            for i, e in enumerate(e_snt):
                # P(f_j, a_j=i | e) = P(a_j=i) P(f_j|e_i, l, m)
                pfj_e += PAj.generate((j, i), e_snt, 0, l, m) * PFj.generate((j, f), (j, i), e_snt, 0, l, m)
            # P(f|z,e) = \prod_j P(f_j|z,e)
            log_pf_e += np.log(pfj_e)
        # \sum_{f,e} P(l)P(m)P(f|e,l,m)
        ll += log_pl + log_pm + log_pf_e
    return - ll / e_corpus.n_sentences()


def log_posterior(e_snt: 'np.array', f_snt: 'np.array', model: ConditionalModel):
    """
    Return the (log) posterior over alignments:
        P(a|f,e) = P(a|f,e)

    :param e_snt: English sentence -- shape: (l,)
    :param f_snt: French sentence -- shape: (m,)
    :param model: a conditional model
    :return: ln(P(a|f,e)) whose shape is (m,l)
    """

    PL, PM, PAj, PFj = model.components

    # observations
    l = e_snt.shape[0]
    m = f_snt.shape[0]

    # Compute joint likelihood: P(a,f|e)

    # P(f,a|e) = \prod_j P(f_j,a_j|e)
    pfa_e = np.zeros((m, l), dtype=ptypes.real)  # shape: (m, l)
    for j, f in enumerate(f_snt):
        for i, e in enumerate(e_snt):
            # where P(f_j,a_j|e) = P(a_j|e)P(f_j|a_j,e)
            pfa_e[j, i] = PAj.generate((j, i), e_snt, 0, l, m) * PFj.generate((j, f), (j, i), e_snt, 0, l, m)

    # Compute posterior probability
    #
    # 1. marginalise alignments
    # p(f_j|e) = \sum_i p(f_j,a_j=i|e)
    log_pfj_e = np.log(pfa_e.sum(1))  # shape: (m,)
    # p(f|e) = \prod_j p(f_j|e)
    log_pf_e = log_pfj_e.sum()  # shape: scalar

    # P(a|f,e) = \prod_j P(a_j|z,f,e)
    # where
    #   P(a_j|f,e) = P(f_j,a_j|e)/P(f_j|,e)
    log_pa_fe = np.log(pfa_e) - log_pfj_e[:, np.newaxis]  # shape: (m, l)

    return log_pa_fe


def posterior(e_snt: 'np.array', f_snt: 'np.array', model: ConditionalModel):
    """Same as log_posterior but exponentiated, i.e. in probability domain"""
    return np.exp(log_posterior(e_snt, f_snt, model))


def EM(e_corpus: Corpus, f_corpus: Corpus, model: ConditionalModel, iterations=5):
    """
    Generative story:

        l ~ P(L)
        m ~ P(M)
        a_j ~ P(A_j | l) for j=1..m
        f_j ~ P(F_j | e_{a_j}) for j=1..m

    :param e_corpus: English data
    :param f_corpus: French data
    :param model: a conditional model
    :param iterations: EM iterations
    """

    PL, PM, PAj, PFj = model.components

    logging.info('Iteration %d Likelihood %f', 0, marginal_likelihood(e_corpus, f_corpus, model))

    for iteration in range(1, iterations + 1):
        # E-step
        for s, (e_snt, f_snt) in enumerate(zip(e_corpus.itersentences(), f_corpus.itersentences())):
            # get the posterior P(a|f,e)
            post_a = posterior(e_snt, f_snt, model)
            l = e_snt.shape[0]
            m = f_snt.shape[0]

            # gather expected counts for (f_j, e_j): p(a_j=i|f,e)
            for j, f in enumerate(f_snt):
                for i, e in enumerate(e_snt):
                    PAj.observe((j, i), e_snt, 0, l, m, post_a[j, i])
                    PFj.observe((j, f), (j, i), e_snt, 0, l, m, post_a[j, i])

        # M-step
        model.update()

        logging.info('Iteration %d Likelihood %f', iteration, marginal_likelihood(e_corpus, f_corpus, model))


def map_decoder(e_corpus: Corpus, f_corpus: Corpus, model: ConditionalModel, callback):
    """

    :param e_corpus: English data
    :param f_corpus: French data
    :param model: components
    :param callback: called for each sentence in the parallel corpus
        callable(s, z, a, p(z|f,e), p(a|z,f,e))
    """

    # E-step
    for s, (e_snt, f_snt) in enumerate(zip(e_corpus.itersentences(), f_corpus.itersentences())):

        log_post_a = log_posterior(e_snt, f_snt, model)

        # Here we get the best path for each cluster
        best_path = log_post_a.argmax(1)  # shape: (m)

        # best posterior probabilities: p(a|z,fe)
        best_posterior = np.array([log_post_a[j, i] for j, i in enumerate(best_path)])

        # communicate the finding
        callback(s, best_path, np.exp(best_posterior))


def get_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    PL = cat.LengthDistribution()
    PM = cat.LengthDistribution()
    PAj = cat.UniformAlignment()
    PFj = cat.BrownLexical(e_corpus.vocab_size(), f_corpus.vocab_size())
    return ConditionalModel(PL, PM, PAj, PFj)


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    model = get_ibm1(e_corpus, f_corpus)

    EM(e_corpus, f_corpus, model, iterations=10)
    from lola.io import print_lola_format
    map_decoder(e_corpus, f_corpus, model,
                partial(print_lola_format,
                        e_corpus=e_corpus,
                        f_corpus=f_corpus,
                        ostream=sys.stdout))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')