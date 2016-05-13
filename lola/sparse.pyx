"""
Auxiliary data structures for sparse representations.

"""

from libcpp.utility cimport pair as cpppair
from cython.operator cimport dereference as deref, preincrement as inc
cimport cython


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

    cpdef float get(self, int key):
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

    cpdef void scale(self, float scalar):
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
        cdef cppmap[int, float].iterator it = self._data.begin()
        while it != self._data.end():
            total += deref(it).second
            inc(it)
        return total + (self._support_size - self._data.size()) * self._base_value

    cpdef float plus_equals(self, int key, float value):
        """
        Adds to the underlying value of an element.
        :param key: an element in the half-open [0, support_size)
        :param value:
        :return: categorical(key)
        """
        cdef cpppair[cppmap[int, float].iterator, bint] result = self._data.insert(cpppair[int, float](key, self._base_value + value))
        if not result.second:
            deref(result.first).second = deref(result.first).second + value
        return deref(result.first).second

    @cython.cdivision(True)
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

    def iternonzero(self):
        return dict(self._data).items()



cdef class CPDTable:
    """
    Organises a fixed number of CPDs,
    each represented as a sparse map of a fixed support size and a base value.
    """

    def __init__(self, size_t n_cpds, size_t support_size, float base_value):
        """
        :param n_cpds: number of CPDs
        :param support_size: support size of each CPD
        :param base_value: base value (common to all elements in all CPDs)
        """
        self._cpds.resize(n_cpds)
        self._support_size = support_size
        self._base_values.resize(n_cpds, base_value)

    def __len__(self):
        return self._cpds.size()

    cpdef float get(self, size_t x, int y):
        """
        Parameter associated with the yth element of the xth CPD.
        """
        cdef cppvector[cppmap[int, float]].iterator cpd_it = self._cpds.begin() + x
        cdef cppmap[int, float].iterator it = deref(cpd_it).find(y)
        if it == deref(cpd_it).end():
            return self._base_values[x]
        else:
            return deref(it).second

    cpdef void scale(self, size_t x, float scalar):
        """
        Scale the xth CPD.
        """
        cdef cppvector[cppmap[int, float]].iterator cpd_it = self._cpds.begin() + x
        cdef cppmap[int, float].iterator it = deref(cpd_it).begin()
        while it != deref(cpd_it).end():
            deref(it).second = deref(it).second * scalar
            inc(it)
        self._base_values[x] *= scalar

    cpdef float sum(self, size_t x):
        """
        Return the sum of values (for mapped and unmapped elements) of the xth CPD.
        """
        cdef cppvector[cppmap[int, float]].iterator cpd_it = self._cpds.begin() + x
        cdef float total = 0.0
        cdef cppmap[int, float].iterator it = deref(cpd_it).begin()
        while it != deref(cpd_it).end():
            total += deref(it).second
            inc(it)
        return total + (self._support_size - deref(cpd_it).size()) * self._base_values[x]

    cpdef float plus_equals(self, size_t x, int y, float value):
        """
        Adds to the underlying value of the yth element of the xth CPD.
        """
        cdef cppvector[cppmap[int, float]].iterator cpd_it = self._cpds.begin() + x
        cdef cpppair[cppmap[int, float].iterator, bint] result = deref(cpd_it).insert(cpppair[int, float](y, self._base_values[x] + value))
        if not result.second:
            deref(result.first).second = deref(result.first).second + value
        return deref(result.first).second

    @cython.cdivision(True)
    cpdef void normalise(self):
        """
        Normalise each CPD independently.
        This checks for 0 mass and skips normalisation in such cases.
        """
        cdef size_t x
        cdef float Z
        for x in range(self._cpds.size()):
            Z = self.sum(x)
            if Z != 0.0:
                self.scale(x, 1.0 / Z)

    def iternonzero(self, size_t x):
        cdef cppvector[cppmap[int, float]].iterator cpd_it = self._cpds.begin() + x
        return dict(deref(cpd_it)).items()