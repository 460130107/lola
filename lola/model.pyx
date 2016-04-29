"""
Authors: - Wilker Aziz
"""


cdef class Model:
    """
    A 0th-order alignment model, that is, alignment links are independent on one another.
    """

    cpdef float pij(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate the probability p(a_j = i , f| e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: p(a_j = i, f | e)
        """
        pass

    cpdef float count(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """
        Increase expected counts for a_j = i.
        :param e_snt: e_0^l
        :param f_snt: f_1^m
        :param i: English position
        :param j: French position
        :param p: posterior probability p(a_j=i|f,e)
        :return: updated count(a_j=i)
        """
        pass

    cpdef normalise(self):
        """
        Normalise expected counts and updated underlying parameters.
        :return:
        """
        pass


cdef class IBM1(Model):
    """
    An IBM1 is a 0th-order model with lexical parameters only.
    """

    def __init__(self, LexicalParameters lex_parameters):
        self._lex_parameters = lex_parameters
        self._lex_counts = LexicalParameters(self._lex_parameters.e_vocab_size(),
                                             self._lex_parameters.f_vocab_size(),
                                             0.0)

    cpdef float pij(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        return self._lex_parameters.get(e_snt[i], f_snt[j])

    cpdef float count(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        return self._lex_counts.plus_equals(e_snt[i], f_snt[j], p)

    cpdef normalise(self):
        self._lex_counts.normalise()
        self._lex_parameters = self._lex_counts
        self._lex_counts = LexicalParameters(self._lex_parameters.e_vocab_size(),
                                             self._lex_parameters.f_vocab_size(),
                                             0.0)
        return self._lex_parameters


# TODO: write IBM2 (standard parameterisation)
# TODO: write IBM2 (Vogel's parameterisation)