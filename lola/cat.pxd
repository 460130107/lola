"""
:Authors: - Wilker Aziz
"""
from lola.ptypes cimport real_t
from lola.sparse cimport CPDTable


# Abstract distributions


cdef class LengthDistribution:

    cpdef real_t generate(self, size_t length)

    cpdef observe(self, size_t length, real_t posterior)

    cpdef update(self)


cdef class AlignmentDistribution:

    cpdef real_t generate(self, tuple aj, e_snt, size_t z, size_t l, size_t m)

    cpdef observe(self, tuple aj, e_snt, size_t z, size_t l, size_t m, real_t posterior)

    cpdef update(self)


cdef class ClusterDistribution:

    cpdef real_t generate(self, size_t z, size_t l, size_t m)

    cpdef observe(self, size_t z, size_t l, size_t m, real_t posterior)

    cpdef update(self)


cdef class TargetDistribution:

    cpdef real_t generate(self, tuple ei, size_t z, size_t l, size_t m)

    cpdef observe(self, tuple ei, size_t z, size_t l, size_t m, real_t posterior)

    cpdef update(self)


cdef class SourceDistribution:

    cpdef real_t generate(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m)

    cpdef observe(self, tuple fj, tuple aj, e_snt, size_t z, size_t l, size_t m, real_t posterior)

    cpdef update(self)


# Implementations


cdef class UniformLength(LengthDistribution):

    cdef size_t _longest


cdef class UniformAlignment(AlignmentDistribution):

   pass


cdef class VogelJump(AlignmentDistribution):

    cdef:
        size_t _longest
        real_t[::1] _cpd, _counts

    cpdef size_t get_jump(self, size_t i, size_t j, size_t l, size_t m)


cdef class ClusterUnigrams(ClusterDistribution):

    cdef real_t[::1] _cpd, _counts


cdef class UnigramMixture(TargetDistribution):

    cdef:
        real_t[:, ::1] _cpds, _counts


cdef class BrownLexical(SourceDistribution):

    cdef:
        real_t[:,::1] _cpds
        CPDTable _counts


cdef class MixtureOfBrownLexical(SourceDistribution):

    cdef:
        list _cpds
        list _counts
