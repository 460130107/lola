"""
A random variable here is a tuple, with a name (str, first element) and perhaps additional fields.
The semantics of additional fields is up to the developer. I am not fixing it.

Examples:
    rv called Z:
        ('Z',)

    rv called Fj where j = 1:
        ('Fj', 1)

An assignment to a rv is a tuple made of the rv (first element) and its value.
The value can be whatever the developer likes, I again not fixing its semantics.

Examples:
    rv Z whose value is 1
        (('Z',), 1)
    rv Fj where j=1 whose value is "dog"
        (('Fj', 1), "dog")
    You can use helper functions:
        assign_rv(make_rv('Z'), 1))
        assign_rv(make_rv('Fj', j), "dog"))

"""
import numpy as np
from collections import defaultdict
import logging
import itertools


def make_rv(name, *args):
    return (name,) + args


def assign_rv(rv, value):
    return rv, value


class Component:
    """
    A locally normalised component.

    Example:

        a Component might be a lexical translation distribution that generates 'Fj' variables.
        then it's rv_name is 'Fj'
    """

    def __init__(self, rv_name: str):
        """
        The random variable it generates. For now this has to be a single random variable.
        Unless you create an rv whose semantics are different.
        TODO: deal with rv_names: list ?

        :param rv_name: str, the name of the random variable.
        """
        self.rv_name = rv_name

    def generate(self, rv: tuple, value, context: dict) -> float:
        """
        Generates a certain assignment of the random variable in context.
        Returns the contribution to the joint distribution.

        :param rv: this is the rv, remember always a tuple, first the name, then the rest
        :param value: value of the rv
        :param context: context containing assignments to other rvs which this component may condition on
        :return: the probability (always locally normalised)
        """
        pass

    def observe(self, rv: tuple, value, context: dict, posterior: float):
        """
        Observe an assignment of the random variable with a certain posterior probability.
        This is typically used in some sort of E-step.

        :param rv: this is the rv, remember always a tuple, first the name, then the rest
        :param value: value of the rv
        :param context: context containing assignments to other rvs which this component may condition on
        :param posterior: posterior probability of relevant latent variables
        """
        pass

    def update(self):
        """
        Update simply maximises the component based on observed expected counts.
        This is typically used in some sort of M-step.
        """
        pass


class UniformLength(Component):
    """
    This is to model P(L) or P(M) for example.
    In this case,
        P(L=l) = 1/longest
        where longest is fixed.
    """

    def __init__(self, longest: int, rv_name):
        super(UniformLength, self).__init__(rv_name)
        self._longest = longest

    def generate(self, rv: tuple, value, context: dict) -> float:
        return 1.0 / self._longest

    def observe(self, rv: tuple, e, context: dict, posterior: float):
        pass

    def update(self):
        pass


class UniformAlignment(Component):
    """
    Models P(A_j|L,M) using a constant, i.e.
        P(A_j=i|L=l,M=m) = 1.0 / l

    Ideal for IBM1.
    """

    def __init__(self, rv_name='Aj', length_name='L'):
        super(UniformAlignment, self).__init__(rv_name)
        self._length_name = length_name

    def generate(self, rv: tuple, value, context: dict) -> float:
        l = context[make_rv(self._length_name)]
        return 1.0 / l

    def observe(self, rv: tuple, e, context: dict, posterior: float):
        pass

    def update(self):
        pass


class VogelJump(Component):
    """
    Models P(Aj|L, M) using a categorical over jump values, where we know a priori the maximum jump, i.e.

        P(Aj=i|L=l,M=m) = Cat(jump(i, j, l, m))

    Ideal for IBM2
    """

    def __init__(self, longest, rv_name='Aj', l_name='L', m_name='M'):
        super(VogelJump, self).__init__(rv_name)
        self._l_name = l_name
        self._m_name = m_name
        self._longest = longest
        self._cpd = np.full(longest * 2, 0.5 / longest, dtype=float)
        self._counts = np.zeros(longest * 2, dtype=float)

    def get_jump(self, rv: tuple, i: int, context: dict):
        j = rv[1]  # rv[0] is the name, i.e. 'Aj', rv[1] in this case is the value of j
        l = context[make_rv(self._l_name)]
        m = context[make_rv(self._m_name)]
        jump = int(i - np.floor((j + 1) * float(l) / m))
        if jump < -self._longest:
            jump = - self._longest
        elif jump >= self._longest:
            jump = self._longest - 1
        return jump

    def generate(self, rv: tuple, value, context: dict) -> float:
        jump = self.get_jump(rv, value, context)
        d = self._longest + jump
        return self._cpd[d]

    def observe(self, rv: tuple, value, context: dict, posterior: float):
        jump = self.get_jump(rv, value, context)
        d = self._longest + jump
        self._counts[d] += posterior

    def update(self):
        self._counts /= self._counts.sum()
        self._cpd = self._counts
        self._counts = np.zeros(self._longest * 2, dtype=float)


class UniformMixture(Component):
    """
    This a dummy mixture of uniform components.
    """

    def __init__(self, n_components, vocab_size, rv_name='Ei', comp_name='Z'):
        super(UnigramMixture, self).__init__(rv_name)
        self._vocab_size = vocab_size
        self._comp_name = comp_name

    def generate(self, rv: tuple, e, context: dict) -> float:
        return 1.0 / self._vocab_size

    def observe(self, rv: tuple, e, context: dict, posterior: float):
        pass

    def update(self):
        pass


class UnigramMixture(Component):
    """
    Mixture of n unigram distributions.

    Example,
        we can model P(Ei|Z) with this component,
        for each value of Z (up to n values) we have one categorical over
        outcomes of Ei

    Ideal for the cluster/joint IBM model.
    """

    def __init__(self, n_components, vocab_size, alpha=0.0, rv_name='Ei', comp_name='Z',
                 rng=np.random.RandomState(1234)):
        super(UnigramMixture, self).__init__(rv_name)
        if alpha > 0.0:  # sample CPDs from a symmetric Dirichlet(alpha)
            self._cpds = rng.dirichlet(np.full(vocab_size, alpha), n_components)
        else:
            self._cpds = np.full((n_components, vocab_size), 1.0 / vocab_size, dtype=float)
        self._counts = np.zeros((n_components, vocab_size), dtype=float)
        self._comp_rv = make_rv(comp_name)

    def generate(self, rv: tuple, e, context: dict) -> float:
        z = context[self._comp_rv]
        return self._cpds[z, e]

    def observe(self, rv: tuple, e, context: dict, posterior: float):
        z = context[self._comp_rv]
        self._counts[z, e] += posterior

    def update(self):
        normalisers = self._counts.sum(1)
        self._counts /= normalisers[:, np.newaxis]
        self._cpds = self._counts
        self._counts = np.zeros(self._counts.shape, dtype=float)


class BrownLexical(Component):
    """
    Models P(Fj|E_Aj) with a categorical distribution per assingment of English word.

    Ideal for IBM1.
    """

    def __init__(self, e_vocab_size, f_vocab_size, rv_name='Fj', e_name='Ei', a_name='Aj'):
        super(BrownLexical, self).__init__(rv_name)
        self._cpds = np.full((e_vocab_size, f_vocab_size), 1.0 / f_vocab_size, dtype=float)
        self._counts = np.zeros((e_vocab_size, f_vocab_size), dtype=float)
        self._e_name = e_name
        self._a_name = a_name

    def generate(self, rv: tuple, f, context: dict) -> float:
        j = rv[1]
        aj = context[(self._a_name, j)]  # looks for an assignment to ('Aj', j)
        e = context[(self._e_name, aj)]  # looks for an assignment to ('Ei', aj)
        return self._cpds[e, f]

    def observe(self, rv: tuple, f, context: dict, posterior: float):
        j = rv[1]
        aj = context[(self._a_name, j)]
        e = context[(self._e_name, aj)]
        self._counts[e, f] += posterior

    def update(self):
        normalisers = self._counts.sum(1)
        self._counts /= normalisers[:, np.newaxis]
        self._cpds = self._counts
        self._counts = np.zeros(self._counts.shape, dtype=float)


class BrownLexicalZ(Component):
    """
    Models P(Fj|E_Aj, Z) with one categorical per pair (English word, cluster).
    """

    def __init__(self, n_clusters, e_vocab_size, f_vocab_size,
                 rv_name='Fj', e_name='Ei', a_name='Aj', z_name='Z'):
        # TODO: add an alpha parameter to initialise with Dirichlet samples
        super(BrownLexicalZ, self).__init__(rv_name)
        self._cpds = np.full((n_clusters, e_vocab_size, f_vocab_size), 1.0 / f_vocab_size, dtype=float)
        self._counts = np.zeros((n_clusters, e_vocab_size, f_vocab_size), dtype=float)
        self._e_name = e_name
        self._a_name = a_name
        self._z_rv = make_rv(z_name)

    def generate(self, rv: tuple, f, context: dict) -> float:
        j = rv[1]
        aj = context[make_rv(self._a_name, j)]
        e = context[make_rv(self._e_name, aj)]
        z = context[self._z_rv]
        return self._cpds[z, e, f]

    def observe(self, rv: tuple, f, context: dict, posterior: float):
        j = rv[1]
        aj = context[make_rv(self._a_name, j)]
        e = context[make_rv(self._e_name, aj)]
        z = context[self._z_rv]
        self._counts[z, e, f] += posterior

    def update(self):
        normalisers = self._counts.sum(2)
        self._counts /= normalisers[:, :, np.newaxis]
        self._cpds = self._counts
        self._counts = np.zeros(self._counts.shape, dtype=float)


class Model:
    """
    A container for generative components.
    """

    def __init__(self, components):
        self._components = tuple(components)
        self._comps_by_rv = defaultdict(list)
        for comp in self._components:
            self._comps_by_rv[comp.rv_name].append(comp)
        for rv_name, comps in self._comps_by_rv.items():
            if len(comps) > 1:
                raise ValueError('In a Bayesian network variables should never be generated twice: %s' % rv_name)

    def __iter__(self):
        return iter(self._components)

    def generate_rv(self, rv: tuple, value, context: dict, state: dict) -> float:
        """
        Generates a variable returning its factor.
        The context is updated at the end of the method (after generation).
        :param rv: a single rv
        :param value: its assignment
        :param context: the context in which it is generated
        :param state: where we store the rv assignment after generation
        :return: probability
        """
        factor = 1.0
        for comp in self._comps_by_rv.get(rv[0], []):  # we filter by rv's name
            factor *= comp.generate(rv, value, context)
        state[rv] = value
        return factor

    def observe_rv(self, rv: tuple, value, context: dict, state: dict, posterior: float):
        """
        Observe a variable updating sufficient statistics.
        The context is updated at the end of the method (after observation).

        :param rv: a single rv
        :param value: its assignment
        :param context: the context in which it is generated
        :param state: where we store the rv assignment after generation
        :param posterior: the posterior over latent variables
        """
        for comp in self._comps_by_rv.get(rv[0], []):
            comp.observe(rv, value, context, posterior)
        state[rv] = value

    def generate(self, predictions: list, context: dict) -> (float, dict):
        """

        :param predictions: an ordered list of rv assingments, i.e., rvs paired with their values
            rvs get generated in the given order.
        :param context: context in which they get generated
        :return: probability, state (which in this case includes the context)
        """
        state = dict(context)
        factor = 1.0
        for rv, value in predictions:
            # update components associated with an RV
            for comp in self._comps_by_rv.get(rv[0], []):
                factor *= comp.generate(rv, value, state)
            state[rv] = value
        return factor, state

    def observe(self, predictions: list, context: dict, posterior: float) -> dict:
        """

        :param predictions: an ordered list of rv assingments, i.e., rvs paired with their values
            rvs get generated in the given order.
        :param context: context in which they get generated
        :param posterior: posterior over latent variables
        :return: state (which in this case includes the context)
        """
        state = dict(context)
        for rv, value in predictions:
            # update components associated with an RV
            for comp in self._comps_by_rv.get(rv[0], []):
                comp.observe(rv, value, state, posterior)
            state[rv] = value
        return state

    def update(self):
        for comp in self._components:
            comp.update()


from lola.corpus import Corpus


def marginal_likelihood(e_corpus: Corpus, f_corpus: Corpus, model: Model, n_clusters):
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
                    p, _ = model.generate(predictions, state)
                    pj += p
                log_pf += np.log(pj)
            ll_s = np.logaddexp(ll_s, log_pz + log_pe + log_pf)
        ll += log_pl + log_pm + ll_s
    return - ll / e_corpus.n_sentences()


def zero_order_joint_model(e_corpus: Corpus, f_corpus: Corpus, model: Model, iterations=5, n_clusters=1):
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
        # E-step
        for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):

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
                        pf_zae[z, j, i], _ = model.generate(predictions, state)

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
                for j, f in enumerate(f_snt):
                    for i, e in enumerate(e_snt):
                        predictions = [(make_rv('Aj', j), i),
                                       (make_rv('Fj', j), f)]
                        model.observe(predictions, state, np.exp(log_post_a[j, i]))

        # M-step
        model.update()

        logging.info('Iteration %d Likelihood %f', iteration, marginal_likelihood(e_corpus, f_corpus, model, n_clusters))


def get_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    components = [UniformAlignment(),
                  BrownLexical(e_corpus.vocab_size(),
                               f_corpus.vocab_size())]
    return Model(components)


def get_ibm2(e_corpus: Corpus, f_corpus: Corpus, ibm1: Model=None):
    if ibm1 is None:
        components = [VogelJump(e_corpus.max_len()),
                      BrownLexical(e_corpus.vocab_size(),
                                   f_corpus.vocab_size())]
        return Model(components)
    else:
        components = list(ibm1)
        components[0] = VogelJump(e_corpus.max_len())  # replace the alignment component
        return Model(components)


def get_joint_ibm1(e_corpus: Corpus, f_corpus: Corpus,
                   n_clusters: int = 1, alpha: float = 1.0):
    components = [UniformLength(e_corpus.max_len(), 'L'),
                  UniformLength(f_corpus.max_len(), 'M'),
                  UniformAlignment(),
                  BrownLexical(e_corpus.vocab_size(),
                               f_corpus.vocab_size()),
                  UnigramMixture(n_clusters,
                                 e_corpus.vocab_size(),
                                 alpha=alpha)]
    return Model(components)


def get_joint_zibm1(e_corpus: Corpus, f_corpus: Corpus,
                    n_clusters: int = 1, alpha: float = 1.0):
    components = [UniformLength(e_corpus.max_len(), 'L'),  # P(L=l) = 1/e_max_len
                  UniformLength(f_corpus.max_len(), 'M'),  # P(M=m) = 1/f_max_len
                  UniformAlignment(),
                  BrownLexicalZ(n_clusters,
                                e_corpus.vocab_size(),
                                f_corpus.vocab_size()),
                  UnigramMixture(n_clusters,
                                 e_corpus.vocab_size(),
                                 alpha=alpha)]
    return Model(components)


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    n_clusters = 1
    model = get_ibm1(e_corpus, f_corpus)
    model = get_joint_ibm1(e_corpus, f_corpus, n_clusters)
    zero_order_joint_model(e_corpus, f_corpus, model, iterations=10, n_clusters=n_clusters)

    # TODO: BrownLexicalPOE




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')

