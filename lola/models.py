"""
:Authors: - Wilker Aziz
"""
import logging
import numpy as np
import lola.cat as cat
from lola.pgm import DirectedModel
from lola.pgm import make_rv
from lola.corpus import Corpus


def marginal_likelihood(e_corpus: Corpus, f_corpus: Corpus, model: DirectedModel, n_clusters):
    ll = 0.0
    for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
        # observations
        context = dict()
        l = e_snt.shape[0]
        m = f_snt.shape[0]
        log_pl = np.log(model.generate_rv(make_rv('L'), l, context, context))
        log_pm = np.log(model.generate_rv(make_rv('M'), m, context, context))
        # 0-order alignments
        ll_s = -np.inf  # contribution of this sentence
        for z in range(n_clusters):
            state = dict(context)
            # contribution of the cluster
            log_pz = np.log(model.generate_rv(make_rv('Z'), z, state, state))
            # compute the contribution of the entire English sentence
            log_pe = 0.0
            for i, e in enumerate(e_snt):
                log_pe += np.log(model.generate_rv(make_rv('Ei', i), e, state, state))
            log_pf = 0.0
            for j, f in enumerate(f_snt):
                pj = 0.0  # contribution of this French word
                for i, e in enumerate(e_snt):
                    predictions = [(make_rv('Aj', j), i),
                                   (make_rv('Fj', j), f)]
                    p, _ = model.generate_rvs(predictions, state)
                    pj += p
                log_pf += np.log(pj)
            ll_s = np.logaddexp(ll_s, log_pz + log_pe + log_pf)
        ll += log_pl + log_pm + ll_s
    return - ll / e_corpus.n_sentences()


def zero_order_joint_model(e_corpus: Corpus, f_corpus: Corpus, model: DirectedModel, iterations=5, n_clusters=1):
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

    logging.info('Iteration %d Likelihood %f', 0, marginal_likelihood(e_corpus, f_corpus, model, n_clusters))

    for iteration in range(1, iterations + 1):
        print(iteration)
        # E-step
        for s, (e_snt, f_snt) in enumerate(zip(e_corpus.itersentences(), f_corpus.itersentences())):
            print('s=%d' % s)
            # observations
            context = {}
            l = e_snt.shape[0]
            m = f_snt.shape[0]
            pl = model.generate_rv(make_rv('L'), l, context, context)
            pm = model.generate_rv(make_rv('M'), m, context, context)

            # compute joint likelihood
            pf_zae = np.ones((n_clusters, m, l), dtype=float)  # p(f|z,a,e)
            log_pze = np.zeros(n_clusters, dtype=float)  # p(z,e) = p(z)p(e|z)
            for z in range(n_clusters):
                state = dict(context)
                log_pz = np.log(model.generate_rv(make_rv('Z'), z, state, state))
                log_pe = 0.0
                for i, e in enumerate(e_snt):
                    log_pe += np.log(model.generate_rv(make_rv('Ei', i), e, state, state))
                log_pze[z] = log_pz + log_pe
                for j, f in enumerate(f_snt):
                    for i, e in enumerate(e_snt):
                        predictions = [(make_rv('Aj', j), i),
                                       (make_rv('Fj', j), f)]
                        pf_zae[z, j, i], _ = model.generate_rvs(predictions, state)

            # gather expected observations based on posterior
            log_pfj_ez = np.log(pf_zae.sum(2))  # p(f_j|e,z) = \sum_i p(f_j,a_j=i|e,z)
            # p(f|e,z) = \sum_a p(f,a|e, z) = \prod_j p(f_j|e,z)
            log_pf_ez = log_pfj_ez.sum(1)  # prod_j p(f_j|e,z)
            log_marginal = np.logaddexp.reduce(log_pze + log_pf_ez)  # p(f,e) = sum_z,a p(f,e,z,a)
            post_z = np.exp(log_pze + log_pf_ez - log_marginal)
            for z in range(n_clusters):
                state = dict(context)
                # gather expected count for z: p(z|f, e)
                model.observe_rv(make_rv('Z'), z, state, state, post_z[z])
                # gather expected counts for (z, e_i): p(z|f, e)
                for i, e in enumerate(e_snt):
                    model.observe_rv(make_rv('Ei', i), e, state, state, post_z[z])
                # gather expected counts for (f_j, e_j): p(a_j=i|f,e,z)
                log_post_a = np.log(pf_zae[z]) - log_pfj_ez[z][:, np.newaxis]
                print(log_post_a)
                print()
                for j, f in enumerate(f_snt):
                    for i, e in enumerate(e_snt):
                        predictions = [(make_rv('Aj', j), i),
                                       (make_rv('Fj', j), f)]
                        model.observe_rvs(predictions, state, np.exp(log_post_a[j, i]))

        # M-step
        model.update()

        logging.info('Iteration %d Likelihood %f', iteration, marginal_likelihood(e_corpus, f_corpus, model, n_clusters))


def get_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    components = [cat.UniformAlignment(),
                  cat.BrownLexical(e_corpus.vocab_size(),
                               f_corpus.vocab_size())]
    return DirectedModel(components)


def get_ibm2(e_corpus: Corpus, f_corpus: Corpus, ibm1: DirectedModel=None):
    if ibm1 is None:
        components = [cat.VogelJump(e_corpus.max_len()),
                      cat.BrownLexical(e_corpus.vocab_size(),
                                   f_corpus.vocab_size())]
        return DirectedModel(components)
    else:
        components = list(ibm1)
        components[0] = cat.VogelJump(e_corpus.max_len())  # replace the alignment component
        return DirectedModel(components)


def get_joint_ibm1(e_corpus: Corpus, f_corpus: Corpus,
                   n_clusters: int = 1, alpha: float = 1.0):
    components = [cat.UniformLength(e_corpus.max_len(), 'L'),
                  cat.UniformLength(f_corpus.max_len(), 'M'),
                  cat.UniformAlignment(),
                  cat.BrownLexical(e_corpus.vocab_size(),
                               f_corpus.vocab_size()),
                  cat.UnigramMixture(n_clusters,
                                 e_corpus.vocab_size(),
                                 alpha=alpha)]
    return DirectedModel(components)


def get_joint_zibm1(e_corpus: Corpus, f_corpus: Corpus,
                    n_clusters: int = 1, alpha: float = 1.0):
    components = [cat.UniformLength(e_corpus.max_len(), 'L'),  # P(L=l) = 1/e_max_len
                  cat.UniformLength(f_corpus.max_len(), 'M'),  # P(M=m) = 1/f_max_len
                  cat.UniformAlignment(),
                  cat.MixtureOfBrownLex(n_clusters,
                                        e_corpus.vocab_size(),
                                        f_corpus.vocab_size()),
                  cat.UnigramMixture(n_clusters,
                                 e_corpus.vocab_size(),
                                 alpha=alpha)]
    return DirectedModel(components)


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    n_clusters = 1
    model = get_ibm1(e_corpus, f_corpus)
    #model = get_joint_ibm1(e_corpus, f_corpus, n_clusters)
    zero_order_joint_model(e_corpus, f_corpus, model, iterations=10, n_clusters=n_clusters)

    # TODO: BrownLexicalPOE

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')
