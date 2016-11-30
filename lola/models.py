"""
:Authors: - Wilker Aziz
"""
import numpy as np
import logging
from lola.corpus import Corpus
import lola.cat as cat
import lola.ptypes as ptypes
import sys
from functools import partial


class Model:
    def __init__(self, PL: cat.LengthDistribution,
                 PM: cat.LengthDistribution,
                 PZ: cat.ClusterDistribution,
                 PEi: cat.TargetDistribution,
                 PAj: cat.AlignmentDistribution,
                 PFj: cat.SourceDistribution):
        self.PL = PL
        self.PM = PM
        self.PZ = PZ
        self.PEi = PEi
        self.PAj = PAj
        self.PFj = PFj
        self.components = tuple([PL, PM, PZ, PEi, PAj, PFj])

    def update(self):
        for comp in self.components:
            comp.update()


def marginal_likelihood(e_corpus: Corpus, f_corpus: Corpus, model: Model):

    PL, PM, PZ, PEi, PAj, PFj = model.components
    n_clusters = PZ.n_clusters
    ll = 0.0
    for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
        # observations
        l = e_snt.shape[0]
        m = f_snt.shape[0]
        log_pl = np.log(PL.generate(l))
        log_pm = np.log(PM.generate(m))
        # 0-order alignments
        # P(f,e) = \sum_z P(z) P(e|z) P(f|z,e)
        log_pfe = -np.inf  # contribution of this sentence
        for z in range(n_clusters):
            # contribution of the cluster
            log_pz = np.log(PZ.generate(z, l, m))
            # compute the contribution of the entire English sentence
            log_pe_z = 0.0
            # P(e|z) = \prod_i P(e_i|z)
            for i, e in enumerate(e_snt):
                log_pe_z += np.log(PEi.generate((i, e), z, l, m))

            # P(f|z,e) = \prod_j P(f_j|z,e)
            #          = \prod_j \sum_i P(f_j,a_j=i|z,e)
            log_pf_ze = 0.0
            for j, f in enumerate(f_snt):
                # P(f_j|z,e) = \sum_i P(f_j,a_j=i|z,e)
                pfj_ze = 0.0  # contribution of this French word
                for i, e in enumerate(e_snt):
                    pfj_ze += PAj.generate((j, i), e_snt, z, l, m) * PFj.generate((j, f), (j, i), e_snt, z, l, m)
                # P(f|z,e) = \prod_j P(f_j|z,e)
                log_pf_ze += np.log(pfj_ze)
            # \sum_z P(z) P(e|z) P(f|z,e)
            log_pfe = np.logaddexp(log_pfe, log_pz + log_pe_z + log_pf_ze)
        # \sum_{f,e} P(l)P(m)P(f,e|l,m)
        ll += log_pl + log_pm + log_pfe
    return - ll / e_corpus.n_sentences()


def log_posterior(e_snt: 'np.array', f_snt: 'np.array', model: Model):
    """
    Return the factorised (log) posterior over latent variables:
        P(z,a|f,e) = P(z|f,e) P(a|z,f,e)

    :param e_snt: English sentence -- shape: (l,)
    :param f_snt: French sentence -- shape: (m,)
    :param PL:
    :param PM:
    :param PZ:
    :param PEi:
    :param PAj:
    :param PFj:
    :return: ln(P(z|f,e)) whose shape is (n_clusters,) and ln(P(a|z,f,e)) whose shape is (n_clusters,m,l)
    """

    PL, PM, PZ, PEi, PAj, PFj = model.components
    n_clusters = PZ.n_clusters

    # observations
    l = e_snt.shape[0]
    m = f_snt.shape[0]
    #log_pl = np.log(PL.generate(l))
    #log_pm = np.log(PM.generate(m))

    # Compute joint likelihood: P(z,a,f,e) = P(z,e)P(f,a|z,e)

    # 1. P(z,e) = P(z)P(e|z) = P(z) \prod_i P(e_i|z)
    log_pze = np.zeros(n_clusters, dtype=ptypes.real)  # shape: (n_clusters,)
    # 2. P(f,a|z,e) = \prod_j P(f_j,a_j|z,e)
    pfa_ze = np.zeros((n_clusters, m, l), dtype=ptypes.real)  # shape: (n_clusters, m, l)
    for z in range(n_clusters):
        # P(z)
        log_pz = np.log(PZ.generate(z, l, m))
        # P(e|z) = \prod_i P(e_i|z)
        log_pe_z = 0.0
        for i, e in enumerate(e_snt):
            log_pe_z += np.log(PEi.generate((i, e), z, l, m))
        # P(z,e) = P(z)P(e|z)
        log_pze[z] = log_pz + log_pe_z
        # P(f,a|z,e) = \prod_j P(f_j,a_j|z,e)
        for j, f in enumerate(f_snt):
            for i, e in enumerate(e_snt):
                # where P(f_j,a_j|z,e) = P(a_j|z,e)P(f_j|a_j,z,e)
                pfa_ze[z, j, i] = PAj.generate((j, i), e_snt, z, l, m) * PFj.generate((j, f), (j, i), e_snt, z, l, m)

    # Compute posterior probability
    #
    # 1. marginalise alignments
    # p(f_j|z,e) = \sum_i p(f_j,a_j=i|z,e)
    log_pfj_ze = np.log(pfa_ze.sum(2))  # shape: (n_clusters, m)
    # p(f|z,e) = \sum_a p(f,a|z,e) = \sum_a \prod_j p(f_j, a_j|z,e)
    #          = \prod_j \sum_i p(f_j,a_j=i|z,e)
    #          = \prod_j p(f_j|z,e)
    log_pf_ze = log_pfj_ze.sum(1)  # shape: (n_clusters,)
    # P(z,f,e) = P(z,e) * P(f|z,e)
    log_pzfe = log_pze + log_pf_ze  # shape: (n_clusters,)
    # P(f,e) = \sum_z P(z,f,e)
    log_pfe = np.logaddexp.reduce(log_pzfe)  # shape: scalar
    # P(z|f,e) = P(z,e,f)/P(e,f)
    log_pz_fe = log_pzfe - log_pfe  # shape: (n_clusters,)

    # P(a|z,f,e) = P(z,e)P(f,a|z,e)/P(z,f,e)
    #            = P(z,e)P(f,a|z,e) / [ P(z,e)P(f|z,e) ]
    #            = P(f,a|z,e) / P(f|z,e)
    #            = \prod_j P(f_j,a_j|z,e) / P(f_j|z,e)
    #            = \prod_j P(a_j|z,f,e)
    # where
    #   P(a_j|z,f,e) = P(f_j,a_j|z,e)/P(f_j|z,e)
    log_pa_zfe = np.log(pfa_ze) - log_pfj_ze[:, :, np.newaxis]  # shape: (n_clusters, m, l)

    return log_pz_fe, log_pa_zfe


def posterior(e_snt: 'np.array', f_snt: 'np.array', model: Model):
    """Same as log_posterior but exponentiate, i.e. in probability domain"""
    logpz, logpa = log_posterior(e_snt, f_snt, model)
    return np.exp(logpz), np.exp(logpa)


def EM(e_corpus: Corpus, f_corpus: Corpus, model: Model, iterations=5):
    """
    Generative story:

        l ~ P(L)
        m ~ P(M)
        z ~ P(Z)
        e_i ~ P(E_i | z) for i=1..l
        a_j ~ P(A_j | l) for j=1..m
        f_j ~ P(F_j | e_{a_j}, z) for j=1..m

    Joint distribution:
        P(F,E,A,Z,L,M) = P(L)P(M)P(Z)P(E|Z)P(A|L,M)P(F|E,A,Z,L,M)

    We make the following independence assumptions:

        P(e|z) = prod_i P(e_i|z)
        P(f|e,a,z,l,m) = prod_j P(a_j|l,m)P(f_j|e_{a_j},z)

    The EM algorithm depends on 2 posterior computations:

        [1] P(z|f,e,l,m) = P(z)P(e|z)P(f|e,z)/P(f,e)
        where
            P(e|z) = \prod_i P(e_i|z)
            P(f|e,z) = \sum_a P(f,a|e,z) = \prod_j \sum_i P(a_j=i)P(f_j|e_i,z)
            P(f,e) = \sum_z \sum_a P(f,e,z,a|l,m)
                = \sum_z P(z)P(e|z) P(f|e,z)
                = \sum_z P(z)P(e|z) \prod_j \sum_i P(a_j=i) P(f_j|e_i,z)
        and
        [2] P(a|f,e,z) = P(a,z,f,e,l,m) / P(f,e,z,l,m)
            =    P(z)(e|z)P(a|l,m)P(f|e,a,z)
              ----------------------------------
              \sum_a P(z)(e|z)P(a|l,m)P(f|e,a,z)
            =    P(z)(e|z)P(a|l,m)P(f|e,a,z)
              ----------------------------------
              P(z)(e|z)\sum_a P(a|l,m)P(f|e,a,z)
            =   P(z)(e|z)\prod_j P(a_j|l,m)P(f_j|e_{a_j},z)
              -----------------------------------------------
              P(z)(e|z)\prod_j\sum_i P(a_j=i|l,m)P(f_j|e_i,z)
            = \prod_j     P(a_j|l,m)P(f_j|e_{a_j},z)
                       -------------------------------
                       \sum_i P(a_j=i|l,m)P(f_j|e_i,z)
            = \prod_j P(a_j|f, e, z)
        where
            P(a_j|f,e,z) =    P(a_j|l,m)P(f_j|e_{a_j},z)
                           -------------------------------
                           \sum_i P(a_j=i|l,m)P(f_j|e_i,z)

    Note that the choice of parameterisation is indenpendent of the EM algorithm in this method.
    For example,
        P(a_j|l,m) can be
            * uniform (IBM1)
            * categorical (IBM2)
        P(f_j|e_{a_j}, z) can be
            * categorical and independent of z, i.e. P(f_j|e_{a_j}, z) = P(f_j|e_{a_j})
            * categorical
            * PoE: P(f_j|e_{a_j}, z) \propto P(f_j|e_{a_j}) P(f_j|z)
            * all of the above using MLP or LR instead of categorical distributions
        we can also have P(a_j|l,m)P(f_j|e_{a_j}, z) modelled by a single LR (with MLP-induced features).

    :param e_corpus: English data
    :param f_corpus: French data
    :param model: all components
    :param iterations: EM iterations
    """

    PL, PM, PZ, PEi, PAj, PFj = model.components
    n_clusters = PZ.n_clusters

    logging.info('Iteration %d Likelihood %f', 0, marginal_likelihood(e_corpus, f_corpus, model))

    for iteration in range(1, iterations + 1):
        # E-step
        for s, (e_snt, f_snt) in enumerate(zip(e_corpus.itersentences(), f_corpus.itersentences())):
            # get the factorised posterior: P(z|f,e) and P(a|z,f,e)
            post_z, post_a = posterior(e_snt, f_snt, model)
            l = e_snt.shape[0]
            m = f_snt.shape[0]
            for z in range(n_clusters):
                # gather expected count for z: p(z|f, e)
                PZ.observe(z, l, m, post_z[z])
                # gather expected counts for (z, e_i): p(z|f, e)
                for i, e in enumerate(e_snt):
                    PEi.observe((i, e), z, l, m, post_z[z])
                # gather expected counts for (f_j, e_j): p(a_j=i|f,e,z)
                for j, f in enumerate(f_snt):
                    for i, e in enumerate(e_snt):
                        PAj.observe((j, i), e_snt, z, l, m, post_a[z, j, i])
                        PFj.observe((j, f), (j, i), e_snt, z, l, m, post_a[z, j, i])

        # M-step
        model.update()

        logging.info('Iteration %d Likelihood %f', iteration, marginal_likelihood(e_corpus, f_corpus, model))


def map_decoder(e_corpus: Corpus, f_corpus: Corpus, model: Model, callback):
    """

    :param e_corpus: English data
    :param f_corpus: French data
    :param model: components
    :param callback: called for each sentence in the parallel corpus
        callable(s, z, a, p(z|f,e), p(a|z,f,e))
    """

    n_clusters = model.PZ.n_clusters
    ll = 0.0
    # E-step
    for s, (e_snt, f_snt) in enumerate(zip(e_corpus.itersentences(), f_corpus.itersentences())):

        log_pz_fe, log_post_a = log_posterior(e_snt, f_snt, model)

        # Here we get the best path for each cluster
        best_paths_z = log_post_a.argmax(2)  # shape: (n_clusters, m)

        # Now we find out which path is the best one across clusters
        best_z = 0
        best_log_prob = -np.inf
        for z in range(n_clusters):
            # p(z,a|f,e) = p(z|f,e) p(a|z,f,e)
            path_log_prob = log_pz_fe[z] + np.sum([log_post_a[z, j, i] for j, i in enumerate(best_paths_z[z])])
            if path_log_prob > best_log_prob:  # update if better
                best_log_prob = path_log_prob
                best_z = z

        # best posterior probabilities: p(a|z,fe)
        best_log_pa_zfe = np.array([log_post_a[z, j, i] for j, i in enumerate(best_paths_z[z])])

        # communicate the finding
        callback(s, best_z, best_paths_z[best_z], np.exp(log_pz_fe[best_z]), np.exp(best_log_pa_zfe))


def get_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    PL = cat.LengthDistribution()
    PM = cat.LengthDistribution()
    PZ = cat.ClusterDistribution(1)
    PEi = cat.TargetDistribution()
    PAj = cat.UniformAlignment()
    PFj = cat.BrownLexical(e_corpus.vocab_size(), f_corpus.vocab_size())
    return Model(PL, PM, PZ, PEi, PAj, PFj)


def get_joint_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    PL = cat.LengthDistribution()
    PM = cat.LengthDistribution()
    PZ = cat.ClusterDistribution(1)
    PEi = cat.UnigramMixture(1, e_corpus.vocab_size())
    PAj = cat.UniformAlignment()
    PFj = cat.BrownLexical(e_corpus.vocab_size(), f_corpus.vocab_size())
    return Model(PL, PM, PZ, PEi, PAj, PFj)


def get_joint_ibm1z(e_corpus: Corpus, f_corpus: Corpus, n_clusters=1, cluster_unigrams=True, alpha=1.0):
    PL = cat.LengthDistribution()
    PM = cat.LengthDistribution()
    if not cluster_unigrams:
        PZ = cat.ClusterDistribution(n_clusters)
    else:
        PZ = cat.ClusterUnigrams(n_clusters)
    PEi = cat.UnigramMixture(n_clusters, e_corpus.vocab_size(), alpha)
    PAj = cat.UniformAlignment()
    PFj = cat.MixtureOfBrownLexical(n_clusters, e_corpus.vocab_size(), f_corpus.vocab_size(), alpha)
    return Model(PL, PM, PZ, PEi, PAj, PFj)


def print_map(s: int, z: int, a: 'np.array', pz: float, pa: 'np.array',
              e_corpus: Corpus, f_corpus: Corpus, ostream=sys.stdout):

    e_snt = e_corpus.sentence(s)
    f_snt = f_corpus.sentence(s)
    tokens = []
    for j, (i, p) in enumerate(zip(a, pa)):
        tokens.append('%d:%s|%d:%s|%.2f' % (j + 1, f_corpus.translate(f_snt[j]),
                                    i, e_corpus.translate(e_snt[i]), p))
    print('%d|%.2f ||| %s' % (z, pz, ' '.join(tokens)), file=ostream)


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    model = get_ibm1(e_corpus, f_corpus)

    EM(e_corpus, f_corpus, model, iterations=10)

    map_decoder(e_corpus, f_corpus, model,
                partial(print_map,
                        e_corpus=e_corpus,
                        f_corpus=f_corpus))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')