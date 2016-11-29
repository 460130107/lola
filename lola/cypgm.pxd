"""
A random variable here is a tuple, with a name (str, first element) and perhaps additional fields.
The semantics of additional fields is up to the developer. I am not fixing it.

Examples:
    rv called Z:
        ('Z',)

    rv called Fj where j = 1:
        ('Fj', 1)

An assignment to a rv is a tuple made of the rv (first element) and its value.
The value can be whatever the developer likes, I again not fixing its semantics.

Examples:
    rv Z whose value is 1
        (('Z',), 1)
    rv Fj where j=1 whose value is "dog"
        (('Fj', 1), "dog")
    You can use helper functions:
        assign_rv(make_rv('Z'), 1))
        assign_rv(make_rv('Fj', j), "dog"))

"""

cpdef tuple make_rv(str name, *args)


cpdef tuple assign_rv(tuple rv, value)


cpdef class GenerativeComponent:
    """
    A locally normalised component.

    Example:

        a Component might be a lexical translation distribution that generates 'Fj' variables.
        then it's rv_name is 'Fj'
    """

    cdef readonly str rv_name

    cpdef float generate(self, tuple rv, value, dict context)

    cpdef observe(self, tuple rv, value, dict context, float posterior)

    cpdef update(self)


cpdef class DirectedModel:
    """
    A container for generative components.
    """

    cdef tuple _components
    cdef dict _comps_by_rv

    cpdef float generate_rv(self, tuple rv, value, dict context, dict state)

    cpdef observe_rv(self, tuple rv, value, dict context, dict state, float posterior)

    cpdef tuple generate_rvs(self, list predictions, dict context)

    cpdef dict observe_rvs(self, list predictions, dict context, float posterior)

    cpdef update(self)
