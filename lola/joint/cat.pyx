"""
:Authors: - Wilker Aziz
"""
cimport numpy as np
import numpy as np
from lola.ptypes cimport real_t
import lola.ptypes as ptypes
from lola.sparse cimport CPDTable


# Dummy abstract distributions


cdef class LengthDistribution:

    cpdef real_t generate(self, size_t length):
        return 1.0

    cpdef observe(self, size_t length, real_t posterior):
        pass

    cpdef update(self):
        pass


cdef class AlignmentDistribution:

    cpdef real_t generate(self, tuple aj, e_snt, size_t z, size_t l, size_t m):
        """
        Generate Aj.
        :param aj: pair (jth French position, ith English position)
        :param e_snt: English sentence
        :param z: cluster
        :param l: English length
        :param m: French length
        :return: P(a_j|z, e, l, m)
        """
        return 1.0

    cpdef observe(self, tuple aj, e_snt, size_t z, size_t l, size_t m, real_t posterior):
        """
        Observe Aj.
        :param aj: pair (jth French position, ith English position)
        :param e_snt: English sentence
        :param z: cluster
        :param l: English length
        :param m: French length
        :param posterior: P(a_j|z, f, e, l, m)
        """
        pass

    cpdef update(self):
        pass


cdef class ClusterDistribution:

    def __init__(self, size_t n_clusters):
        self.n_clusters = n_clusters

    cpdef real_t generate(self, size_t z, size_t l, size_t m):
        """

        :param z:
        :param l:
        :param m:
        :return: P(z|l,m)
        """
        return 1.0

    cpdef observe(self, size_t z, size_t l, size_t m, real_t posterior):
        """

        :param z:
        :param l:
        :param m:
        :param posterior: P(z|f,e)
        :return:
        """
        pass

    cpdef update(self):
        pass


cdef class TargetDistribution:

    cpdef real_t generate(self, tuple ei, size_t z, size_t l, size_t m):
        """
        Generate Ei.

        :param ei: a pair (ith English position, ith English word)
        :param z: cluster
        :param l: English length
        :param m: French length
        :return: P(e_i|z,l,m)
        """
        return 1.0

    cpdef observe(self, tuple ei, size_t z, size_t l, size_t m, real_t posterior):
        """
        Observe Ei.

        :param ei: a pair (ith English position, ith English word)
        :param z: cluster
        :param l: English length
        :param m: French length
        :param posterior: P(z|f,e)
        """
        pass

    cpdef update(self):
        pass


cdef class SourceDistribution:

    cpdef real_t generate(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m):
        """
        Generate Fj.

        :param fj: a pair (jth French position, jth French word)
        :param aj: a pair (jth French position, ith English position)
        :param e_snt: English sentence
        :param z: cluster
        :param l: English length
        :param m: French length
        :return: P(f_j|a_j, z, e)
        """
        return 1.0

    cpdef observe(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m, real_t posterior):
        """
        Observe a fractional count for Fj.

        :param fj: a pair (jth French position, jth French word)
        :param aj: a pair (jth French position, ith English position)
        :param e_snt: English sentence
        :param z: cluster
        :param l: English length
        :param m: French length
        :param posterior: P(a_j|f,e)
        """
        pass

    cpdef update(self):
        pass


# Implementations


cdef class UniformLength(LengthDistribution):
    """
    This is to model P(L) or P(M) for example.
    In this case,
        P(L=l) = 1/longest
        where longest is fixed.
    """

    def __init__(self, size_t longest):
        super(UniformLength, self).__init__()
        self._longest = longest

    cpdef real_t generate(self, size_t length):
        return 1.0 / self._longest


cdef class UniformAlignment(AlignmentDistribution):
    """
    Models P(A_j|L,M) using a constant, i.e.
        P(A_j=i|L=l,M=m) = 1.0 / l

    Ideal for IBM1.
    """

    def __init__(self):
        super(UniformAlignment, self).__init__()

    cpdef real_t generate(self, tuple aj, e_snt, size_t z, size_t l, size_t m):
        return 1.0 / l


cdef class VogelJump(AlignmentDistribution):
    """
    Models P(Aj|L, M) using a categorical over jump values, where we know a priori the maximum jump, i.e.

        P(Aj=i|L=l,M=m) = Cat(jump(i, j, l, m))

    Ideal for IBM2
    """

    def __init__(self, size_t longest):
        super(VogelJump, self).__init__()
        self._longest = longest
        self._cpd = np.full(longest * 2, 0.5 / longest, dtype=ptypes.real)
        self._counts = np.zeros(longest * 2, dtype=ptypes.real)

    cpdef size_t get_jump(self, size_t i, size_t j, size_t l, size_t m):
        cdef int jump = int(<int>i - np.floor((j + 1) * float(l) / m))
        if jump < -<int>self._longest:
            jump = - <int>self._longest
        elif jump >= <int>self._longest:
            jump = <int>self._longest - 1
        return <size_t>jump

    cpdef real_t generate(self, tuple aj, e_snt, size_t z, size_t l, size_t m):
        cdef size_t jump = self.get_jump(aj[1], aj[0], l, m)
        cdef size_t d = self._longest + jump
        return self._cpd[d]

    cpdef observe(self, tuple aj, e_snt, size_t z, size_t l, size_t m, real_t posterior):
        cdef size_t jump = self.get_jump(aj[1], aj[0], l, m)
        cdef size_t d = self._longest + jump
        self._counts[d] += posterior

    cpdef update(self):
        self._cpd = np.divide(self._counts, np.sum(self._counts))
        self._counts = np.zeros(self._longest * 2, dtype=ptypes.real)


cdef class ClusterUnigrams(ClusterDistribution):

    def __init__(self, size_t n_clusters,
                 float alpha=0.0, rng=np.random.RandomState(1234)):
        super(ClusterUnigrams, self).__init__(n_clusters)
        if alpha > 0.0:  # sample CPDs from a symmetric Dirichlet(alpha)
            self._cpd = rng.dirichlet(np.full(n_clusters, alpha), 1)
        else:
            self._cpd = np.full(n_clusters, 1.0 / n_clusters, dtype=ptypes.real)
        self._counts = np.zeros(n_clusters, dtype=ptypes.real)

    cpdef real_t generate(self, size_t z, size_t l, size_t m):
        return self._cpd[z]

    cpdef observe(self, size_t z, size_t l, size_t m, real_t posterior):
        self._counts[z] += posterior

    cpdef update(self):
        self._cpd = np.divide(self._counts, np.sum(self._counts))
        self._counts = np.zeros(np.shape(self._counts), dtype=ptypes.real)


cdef class UnigramMixture(TargetDistribution):
    """
    Mixture of n unigram distributions.

    Example,
        we can model P(Ei|Z) with this component,
        for each value of Z (up to n values) we have one categorical over
        outcomes of Ei

    Ideal for the cluster/joint IBM model.
    """

    def __init__(self, size_t n_components, size_t vocab_size,
                 float alpha=0.0, rng=np.random.RandomState(1234)):
        super(UnigramMixture, self).__init__()
        if alpha > 0.0:  # sample CPDs from a symmetric Dirichlet(alpha)
            self._cpds = rng.dirichlet(np.full(vocab_size, alpha), n_components)
        else:
            self._cpds = np.full((n_components, vocab_size), 1.0 / vocab_size, dtype=ptypes.real)
        self._counts = np.zeros((n_components, vocab_size), dtype=ptypes.real)

    cpdef real_t generate(self, tuple ei, size_t z, size_t l, size_t m):
        cdef size_t e = ei[1]
        return self._cpds[z, e]

    cpdef observe(self, tuple ei, size_t z, size_t l, size_t m, real_t posterior):
        cdef size_t e = ei[1]
        self._counts[z, e] += posterior

    cpdef update(self):
        self._cpds = np.divide(self._counts, np.sum(self._counts, 1)[:, np.newaxis])
        self._counts = np.zeros(np.shape(self._counts), dtype=ptypes.real)


cdef class BrownLexical(SourceDistribution):
    """
    Models P(Fj|E_Aj) with a categorical distribution per assingment of English word.

    Ideal for IBM1.
    """

    def __init__(self, size_t e_vocab_size, size_t f_vocab_size,
                 float alpha=0.0, rng=np.random.RandomState(1234)):
        super(BrownLexical, self).__init__()
        if alpha > 0.0:  # sample CPDs from a symmetric Dirichlet(alpha)
            self._cpds = rng.dirichlet(np.full(f_vocab_size, alpha), e_vocab_size)
        else:  # uniform CPDs
            self._cpds = np.full((e_vocab_size, f_vocab_size), 1.0 / f_vocab_size, dtype=ptypes.real)
        self._counts = CPDTable(e_vocab_size, f_vocab_size, 0.0)

    cpdef real_t generate(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m):
        cdef:
            # j, f = fj
            size_t f = fj[1]
            # j, i = aj
            size_t i = aj[1]
            size_t e = e_snt[i]
        return self._cpds[e, f]

    cpdef observe(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m, real_t posterior):
        cdef:
            size_t f = fj[1]
            size_t i = aj[1]
            size_t e = e_snt[i]
        self._counts.plus_equals(e, f, posterior)

    cpdef update(self):
        cdef size_t x, y
        x, y = self._counts.shape
        self._counts.normalise()
        self._cpds = self._counts.make_dense()
        self._counts = CPDTable(x, y, 0.0)


cdef class MixtureOfBrownLexical(SourceDistribution):
    """
    Models P(Fj|E_Aj, Z) with one categorical per pair (English word, cluster).
    """

    def __init__(self, size_t n_clusters, size_t e_vocab_size, size_t f_vocab_size,
                 float alpha=0.0, rng=np.random.RandomState(1234)):
        super(MixtureOfBrownLexical, self).__init__()
        if alpha > 0.0:
            self._cpds = [rng.dirichlet(np.full(f_vocab_size, alpha), e_vocab_size) for _ in range(n_clusters)]
        else:
            self._cpds = [np.full((e_vocab_size, f_vocab_size), 1.0 / f_vocab_size, dtype=ptypes.real) for _ in range(n_clusters)]
        self._counts = [CPDTable(e_vocab_size, f_vocab_size, 0.0) for _ in range(n_clusters)]

    cpdef real_t generate(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m):
        cdef:
            size_t f = fj[1]
            size_t i = aj[1]
            size_t e = e_snt[i]
        return self._cpds[z][e, f]

    cpdef observe(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m, real_t posterior):
        cdef:
            size_t f = fj[1]
            size_t i = aj[1]
            size_t e = e_snt[i]
        self._counts[z].plus_equals(e, f, posterior)

    cpdef update(self):
        cdef size_t nZ, vE, vF, z
        nZ = len(self._cpds)
        vE, vF = np.shape(self._cpds[0])
        for z in range(nZ):
            self._counts[z].normalise()
            self._cpds[z] = self._counts[z].make_dense()
            self._counts[z] = CPDTable(vE, vF, 0.0)

