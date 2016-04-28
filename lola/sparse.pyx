"""
:Authors: - Wilker Aziz
"""

from libcpp.map cimport map as cppmap
#from libcpp.unordered_map cimport unordered_map as cppmap
from libcpp.utility cimport pair as cpppair
from cython.operator cimport dereference as deref, preincrement as inc

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cdef class SparseCategorical:
    """
    A sparse representation for a categorical distribution.
    We allow 0 to be represented by a non-zero value.

    * does not check for out-of-bound 
    """

    def __init__(self, int support_size, float zero=0.0):
        self._support_size = support_size
        self._zero = zero

    cpdef size_t support_size(self):
        return self._support_size

    cpdef size_t n_represented(self):
        return self._data.size()

    cpdef get(self, int key):
        cdef cppmap[int, float].iterator it = self._data.find(key)
        if it == self._data.end():
            return self._zero
        else:
            return deref(it).second

    cpdef scale(self, float scalar):
        cdef cppmap[int, float].iterator it = self._data.begin()
        while it != self._data.end():
            deref(it).second = deref(it).second * scalar
            inc(it)
        self._zero *= scalar

    cpdef float sum(self):
        cdef cppmap[int, float].iterator it = self._data.begin()
        cdef float total = 0.0
        while it != self._data.end():
            total += deref(it).second
            inc(it)
        return total + (self._support_size - self._data.size()) * self._zero

    cpdef float acc(self, int key, float value):
        cdef cpppair[cppmap[int, float].iterator, bint] result = self._data.insert((key, self._zero + value))
        if not result.second:
            deref(result.first).second = deref(result.first).second + value
        return deref(result.first).second

    cpdef float normalise(self):
        cdef float Z = self.sum()
        if Z != 0.0:
            self.scale(1.0/Z)
        return Z

    def __str__(self):
        return ' '.join(['{0}:{1}'.format(k,v) for k, v in self._data])
