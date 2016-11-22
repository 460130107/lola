from libcpp.unordered_map cimport unordered_map as cppmap
from libcpp.vector cimport vector as cppvector


cdef class SparseCategorical:

    cdef cppmap[int, float] _data
    cdef float _base_value
    cdef size_t _support_size

    cpdef float base_value(self)

    cpdef size_t support_size(self)

    cpdef size_t n_represented(self)

    cpdef float get(self, int key)

    cpdef void scale(self, float scalar)

    cpdef float sum(self)

    cpdef float plus_equals(self, int key, float value)

    cpdef float normalise(self)


cdef class CPDTable:

    cdef:
        cppvector[cppmap[int, float]] _cpds
        cppvector[float] _base_values
        size_t _support_size

    cpdef float get(self, size_t x, int y)

    cpdef void scale(self, size_t x, float scalar)

    cpdef float sum(self, size_t x)

    cpdef float plus_equals(self, size_t x, int y, float value)

    cpdef void normalise(self)