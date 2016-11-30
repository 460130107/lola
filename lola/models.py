"""
:Authors: - Wilker Aziz
"""
import numpy as np
import logging
from lola.corpus import Corpus
import lola.cat as cat


def marginal_likelihood(e_corpus: Corpus, f_corpus: Corpus,
                        PL: cat.LengthDistribution,
                        PM: cat.LengthDistribution,
                        PZ: cat.ClusterDistribution,
                        PEi: cat.TargetDistribution,
                        PAj: cat.AlignmentDistribution,
                        PFj: cat.SourceDistribution,
                        n_clusters):
    ll = 0.0
    for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
        # observations
        context = dict()
        l = e_snt.shape[0]
        m = f_snt.shape[0]
        log_pl = np.log(PL.generate(l))
        log_pm = np.log(PM.generate(m))
        # 0-order alignments
        # P(f,e) = \sum_z P(z) P(e|z) P(f|z,e)
        log_pfe = -np.inf  # contribution of this sentence
        for z in range(n_clusters):
            state = dict(context)
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


def zero_order_joint_model(e_corpus: Corpus, f_corpus: Corpus,
                           PL: cat.LengthDistribution,
                           PM: cat.LengthDistribution,
                           PZ: cat.ClusterDistribution,
                           PEi: cat.TargetDistribution,
                           PAj: cat.AlignmentDistribution,
                           PFj: cat.SourceDistribution,
                           iterations=5, n_clusters=1):
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


    :param e_corpus:
    :param f_corpus:
    :param model:
    :param iterations:
    :param n_clusters:
    :return:
    """

    components = [PL, PM, PZ, PEi, PAj, PFj]


    logging.info('Iteration %d Likelihood %f', 0, marginal_likelihood(e_corpus, f_corpus,
                                                                      PL, PM, PZ, PEi, PAj, PFj,
                                                                      n_clusters))

    for iteration in range(1, iterations + 1):
        # E-step
        for s, (e_snt, f_snt) in enumerate(zip(e_corpus.itersentences(), f_corpus.itersentences())):
            # observations
            l = e_snt.shape[0]
            m = f_snt.shape[0]
            pl = PL.generate(l)
            pm = PM.generate(m)

            # Compute joint likelihood: P(z,a,f,e) = P(z,e)P(f,a|z,e)

            # 1. P(z,e) = P(z)P(e|z) = P(z) \prod_i P(e_i|z)
            log_pze = np.zeros(n_clusters, dtype=float)  # shape: (n_clusters,)
            # 2. P(f,a|z,e) = \prod_j P(f_j,a_j|z,e)
            pfa_ze = np.zeros((n_clusters, m, l), dtype=float)  # shape: (n_clusters, m, l)
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

            # Gather expected observations based on posterior
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
            # P(z|e,f) = P(z,e,f)/P(e,f)
            post_z = np.exp(log_pzfe - log_pfe)  # shape: (n_clusters,)

            #print(pfa_ze)
            #print(log_pfj_ze)
            #print(np.log(pfa_ze) - log_pfj_ze[:,:,np.newaxis])
            #print()

            # P(a|z,f,e) = P(z,e)P(f,a|z,e)/P(z,f,e)
            #            = P(z,e)P(f,a|z,e) / [ P(z,e)P(f|z,e) ]
            #            = P(f,a|z,e) / P(f|z,e)
            #            = \prod_j P(f_j,a_j|z,e) / P(f_j|z,e)
            #            = \prod_j P(a_j|z,f,e)
            # where
            #   P(a_j|z,f,e) = P(f_j,a_j|z,e)/P(f_j|z,e)
            post_a = np.exp(np.log(pfa_ze) - log_pfj_ze[:, :, np.newaxis])  # shape: (n_clusters, m, l)
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
        for comp in components:
            comp.update()

        logging.info('Iteration %d Likelihood %f', iteration, marginal_likelihood(e_corpus, f_corpus,
                                                                          PL, PM, PZ, PEi, PAj, PFj,
                                                                          n_clusters))


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    n_clusters = 1
    PL = cat.LengthDistribution()
    #PL = cat.UniformLength(e_corpus.max_len())
    PM = cat.LengthDistribution()
    #PM = cat.UniformLength(f_corpus.max_len())
    PZ = cat.ClusterDistribution()
    #PZ = cat.ClusterUnigrams(n_clusters)
    PEi = cat.TargetDistribution()
    #PEi = cat.UnigramMixture(n_clusters, e_corpus.vocab_size(), alpha=1.0)
    PAj = cat.UniformAlignment()
    #PAj = cat.VogelJump(e_corpus.max_len())
    PFj = cat.MixtureOfBrownLexical(n_clusters, e_corpus.vocab_size(), f_corpus.vocab_size())

    zero_order_joint_model(e_corpus, f_corpus,
                           PL, PM, PZ, PEi, PAj, PFj,
                           iterations=10, n_clusters=n_clusters)

    # TODO: BrownLexicalPOE

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')