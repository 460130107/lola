"""
:Authors: - Wilker Aziz
"""
import numpy as np
from lola.pgm import RandomVariable
from lola.pgm import GenerativeComponent
from lola.pgm import make_rv
from lola.sparse import CPDTable


class UniformLength(GenerativeComponent):
    """
    This is to model P(L) or P(M) for example.
    In this case,
        P(L=l) = 1/longest
        where longest is fixed.
    """

    def __init__(self, longest: int, rv_name):
        super(UniformLength, self).__init__(rv_name)
        self._longest = longest

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        return 1.0 / self._longest

    def observe(self, rv: RandomVariable, value, context: dict, posterior: float):
        pass

    def update(self):
        pass


class UniformAlignment(GenerativeComponent):
    """
    Models P(A_j|L,M) using a constant, i.e.
        P(A_j=i|L=l,M=m) = 1.0 / l

    Ideal for IBM1.
    """

    def __init__(self, rv_name='Aj', length_name='L'):
        super(UniformAlignment, self).__init__(rv_name)
        self._length_rv = make_rv(length_name)

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        l = context[self._length_rv]
        return 1.0 / l

    def observe(self, rv: RandomVariable, value, context: dict, posterior: float):
        pass

    def update(self):
        pass


class VogelJump(GenerativeComponent):
    """
    Models P(Aj|L, M) using a categorical over jump values, where we know a priori the maximum jump, i.e.

        P(Aj=i|L=l,M=m) = Cat(jump(i, j, l, m))

    Ideal for IBM2
    """

    def __init__(self, longest, rv_name='Aj', l_name='L', m_name='M'):
        super(VogelJump, self).__init__(rv_name)
        self._l_rv = make_rv(l_name)
        self._m_rv = make_rv(m_name)
        self._longest = longest
        self._cpd = np.full(longest * 2, 0.5 / longest, dtype=float)
        self._counts = np.zeros(longest * 2, dtype=float)

    def get_jump(self, rv: RandomVariable, value, context: dict):
        i = value  # type: int
        j = rv[1]  # rv[0] is the name, i.e. 'Aj', rv[1] in this case is the value of j
        l = context[self._l_rv]
        m = context[self._m_rv]
        jump = int(i - np.floor((j + 1) * float(l) / m))
        if jump < -self._longest:
            jump = - self._longest
        elif jump >= self._longest:
            jump = self._longest - 1
        return jump

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        jump = self.get_jump(rv, value, context)
        d = self._longest + jump
        return self._cpd[d]

    def observe(self, rv: RandomVariable, value, context: dict, posterior: float):
        jump = self.get_jump(rv, value, context)
        d = self._longest + jump
        self._counts[d] += posterior

    def update(self):
        self._counts /= self._counts.sum()
        self._cpd = self._counts
        self._counts = np.zeros(self._longest * 2, dtype=float)


class UniformMixture(GenerativeComponent):
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


class UnigramMixture(GenerativeComponent):
    """
    Mixture of n unigram distributions.

    Example,
        we can model P(Ei|Z) with this component,
        for each value of Z (up to n values) we have one categorical over
        outcomes of Ei

    Ideal for the cluster/joint IBM model.
    """

    def __init__(self, n_components, vocab_size, rv_name='Ei', comp_name='Z',
                 alpha=0.0, rng=np.random.RandomState(1234)):
        super(UnigramMixture, self).__init__(rv_name)
        if alpha > 0.0:  # sample CPDs from a symmetric Dirichlet(alpha)
            self._cpds = rng.dirichlet(np.full(vocab_size, alpha), n_components)
        else:
            self._cpds = np.full((n_components, vocab_size), 1.0 / vocab_size, dtype=float)
        self._counts = np.zeros((n_components, vocab_size), dtype=float)
        self._comp_rv = make_rv(comp_name)

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        e = value  # type: int
        z = context[self._comp_rv]
        return self._cpds[z, e]

    def observe(self, rv: RandomVariable, value, context: dict, posterior: float):
        e = value  # type: int
        z = context[self._comp_rv]
        self._counts[z, e] += posterior

    def update(self):
        normalisers = self._counts.sum(1)
        self._counts /= normalisers[:, np.newaxis]
        self._cpds = self._counts
        self._counts = np.zeros(self._counts.shape, dtype=float)


class BrownLexical(GenerativeComponent):
    """
    Models P(Fj|E_Aj) with a categorical distribution per assingment of English word.

    Ideal for IBM1.
    """

    def __init__(self, e_vocab_size, f_vocab_size, rv_name='Fj', e_name='Ei', a_name='Aj',
                 alpha=0.0, rng=np.random.RandomState(1234)):
        super(BrownLexical, self).__init__(rv_name)
        if alpha > 0.0:  # sample CPDs from a symmetric Dirichlet(alpha)
            self._cpds = rng.dirichlet(np.full(f_vocab_size, alpha), e_vocab_size)
        else:  # uniform CPDs
            self._cpds = np.full((e_vocab_size, f_vocab_size), 1.0 / f_vocab_size, dtype=float)
        self._counts = CPDTable(e_vocab_size, f_vocab_size, 0.0)
        self._e_name = e_name
        self._a_name = a_name

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        f = value  # type: int
        j = rv[1]
        aj = context[(self._a_name, j)]  # looks for an assignment to ('Aj', j)
        e = context[(self._e_name, aj)]  # looks for an assignment to ('Ei', aj)
        return self._cpds[e, f]

    def observe(self, rv: RandomVariable, value, context: dict, posterior: float):
        f = value  # type: int
        j = rv[1]
        aj = context[(self._a_name, j)]
        e = context[(self._e_name, aj)]
        self._counts.plus_equals(e, f, posterior)

    def update(self):
        self._counts.normalise()
        self._cpds = self._counts.make_dense()
        x, y = self._counts.shape
        self._counts = CPDTable(x, y, 0.0)


class MixtureOfBrownLex(GenerativeComponent):
    """
    Models P(Fj|E_Aj, Z) with one categorical per pair (English word, cluster).
    """

    def __init__(self, n_clusters, e_vocab_size, f_vocab_size,
                 rv_name='Fj', e_name='Ei', a_name='Aj', z_name='Z',
                 alpha=0.0, rng=np.random.RandomState(1234)):
        super(MixtureOfBrownLex, self).__init__(rv_name)
        if alpha > 0.0:
            self._cpds = rng.dirichlet(np.full(f_vocab_size, alpha), (n_clusters, e_vocab_size))
        else:
            self._cpds = np.full((n_clusters, e_vocab_size, f_vocab_size), 1.0 / f_vocab_size, dtype=float)
        self._counts = [CPDTable(e_vocab_size, f_vocab_size, 0.0) for _ in range(n_clusters)]
        self._e_name = e_name
        self._a_name = a_name
        self._z_rv = make_rv(z_name)

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        f = value  # type: int
        j = rv[1]
        aj = context[(self._a_name, j)]
        e = context[(self._e_name, aj)]
        z = context[self._z_rv]
        return self._cpds[z, e, f]

    def observe(self, rv: tuple, value, context: dict, posterior: float):
        f = value  # type: int
        j = rv[1]
        aj = context[(self._a_name, j)]
        e = context[(self._e_name, aj)]
        z = context[self._z_rv]
        self._counts[z].plus_equals(e, f, posterior)

    def update(self):
        nZ, vE, vF = self._cpds.shape
        self._cpds = np.zeros(self._cpds.shape, dtype=float)
        for z in range(nZ):
            self._counts[z].normalise()
            self._cpds[z] = self._counts[z].make_dense()
            self._counts[z] = CPDTable(vE, vF, 0.0)

