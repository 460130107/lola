"""
Auxiliary data structures for sparse representations.

"""

from libcpp.utility cimport pair as cpppair
from cython.operator cimport dereference as deref, preincrement as inc



cdef class SparseCategorical:
    """
    A sparse representation for a categorical distribution.
    We allow 0 to be represented by a non-zero value (convenient for uniform distributions).

    This implementation does not check for out-of-bound keys.
    """

    def __init__(self, int support_size, float base_value=0.0):
        """
        :param support_size: used in determining the sum of elements.
        :param base_value: value of unmapped elements in the support.
        """
        self._support_size = support_size
        self._base_value = base_value

    cpdef float base_value(self):
        """Return the base value used for unmapped elements."""
        return self._base_value

    cpdef size_t support_size(self):
        """How many elements we expect in the support."""
        return self._support_size

    cpdef size_t n_represented(self):
        """How many elements have been mapped."""
        return self._data.size()

    cpdef get(self, int key):
        """
        Return the value associated with a key (this operation never changes the container).
        It is a logical error to get a value for a key which is not supposed to belong to the support.
        This implementation won't check for limits though.

        :param key: an element in the half-open [0, support_size)
        :return: categorical(key)
        """
        cdef cppmap[int, float].iterator it = self._data.find(key)
        if it == self._data.end():
            return self._base_value
        else:
            return deref(it).second

    cpdef scale(self, float scalar):
        """
        Scales all mapped elements as well as the base value.
        :param scalar: a real number (typically strictly positive)
        """
        cdef cppmap[int, float].iterator it = self._data.begin()
        while it != self._data.end():
            deref(it).second = deref(it).second * scalar
            inc(it)
        self._base_value *= scalar

    cpdef float sum(self):
        """
        Return the sum of values (for mapped and unmapped elements).
        :return: total mass
        """
        cdef float total = 0.0
        cdef int k
        cdef float v
        for k, v in self._data:
            total += v
        return total + (self._support_size - self._data.size()) * self._base_value

    cpdef float plus_equals(self, int key, float value):
        """
        Adds to the underlying value of an element.
        :param key: an element in the half-open [0, support_size)
        :param value:
        :return: categorical(key)
        """
        cdef cpppair[cppmap[int, float].iterator, bint] result = self._data.insert((key, self._base_value + value))
        if not result.second:
            deref(result.first).second = deref(result.first).second + value
        return deref(result.first).second

    cpdef float normalise(self):
        """
        Normalise the elements of the distribution including the base value.
        This checks for 0 mass and skips normalisation in that case.

        :return: total mass before normalisation
        """
        cdef float Z = self.sum()
        if Z != 0.0:
            self.scale(1.0/Z)
        return Z

    def __str__(self):
        cdef str mapped = ' '.join(['{0}:{1}'.format(k,v) for k, v in sorted(dict(self._data).items(), key=lambda pair: pair[0])])
        return 'support=%d base-value=%d mapped=(%s)' % (self._support_size, self._base_value, mapped)