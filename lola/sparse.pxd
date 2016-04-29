"""
Sparse data structures with C++ underlying containers.

:Authors: - Wilker Aziz
"""

#from libcpp.map cimport map as cppmap
from libcpp.unordered_map cimport unordered_map as cppmap


cdef class SparseCategorical:

    cdef cppmap[int, float] _data
    cdef float _base_value
    cdef size_t _support_size

    cpdef float base_value(self)
    
    cpdef size_t support_size(self)

    cpdef size_t n_represented(self)

    cpdef get(self, int key)

    cpdef scale(self, float scalar)

    cpdef float sum(self)

    cpdef float plus_equals(self, int key, float value)
    
    cpdef float normalise(self)


cdef class LexicalParameters:

    cdef list _cpds

    cpdef size_t e_vocab_size(self)

    cpdef size_t f_vocab_size(self)

    cpdef float get(self, int e, int f)

    cpdef float plus_equals(self, int e, int f, float value)

    cpdef SparseCategorical row(self, int e)

    cpdef normalise(self)

