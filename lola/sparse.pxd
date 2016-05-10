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