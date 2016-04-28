"""
Sparse data structures with C++ underlying containers.

:Authors: - Wilker Aziz
"""

from libcpp.map cimport map as cppmap


cdef class SparseCategorical:

    cdef cppmap[int, float] _data
    cdef float _zero
    cdef int _support_size
    
    cpdef size_t support_size(self)

    cpdef size_t n_represented(self)

    cpdef get(self, int key)

    cpdef scale(self, float scalar)

    cpdef float sum(self)

    cpdef float acc(self, int key, float value)
    
    cpdef float normalise(self)

