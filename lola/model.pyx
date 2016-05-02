import numpy as np

cdef class Model:
    """
    A 0th-order alignment model, that is, alignment links are independent on one another.
    """

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: p(a_j = i, f | e)
        """
        pass

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i | f, e) up to a normalisation constant.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: Z * p(a_j = i | f, e)
        """
        pass

    cpdef initialise(self, dict initialiser):
        """
        Some models can be initialised from other models parameters.
        """
        pass


cdef class SufficientStatistics:
    """
    This is used to gather sufficient statistics under a certain model.
    The typical use is to accumulated expected counts from (potential) observations.

    This object also knows how to construct a new model based on up-to-date statistics.
    """

    cpdef observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """
        Account for a potential observation.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :param p: probability of observation, i.e., normalised posterior p(a_j=i | f, e)
        """
        pass

    cpdef Model make_model(self):
        """
        Construct a new model based on sufficient statistics and reset statistics to zero.
        """
        pass


cdef class IBM1ExpectedCounts(SufficientStatistics):

    def __init__(self, size_t e_vocab_size, f_vocab_size):
        self._lex_counts = LexicalParameters(e_vocab_size,
                                             f_vocab_size,
                                             0.0)

    cpdef observation(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j, float p):
        """
        Account for a potential observation:
            * f_j generated from e_i

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :param p: probability of observation, i.e., normalised posterior p(a_j=i | f, e)
        """
        self._lex_counts.plus_equals(e_snt[i], f_snt[j], p)

    cpdef Model make_model(self):
        """
        Return a model whose parameters are the normalised expected lexical counts.
        This also resets the sufficient statistics to zero.
        :return: an instance of IBM1
        """
        self._lex_counts.normalise()
        cdef Model model = IBM1(self._lex_counts)
        self._lex_counts = LexicalParameters(self._lex_counts.e_vocab_size(),
                                             self._lex_counts.f_vocab_size(),
                                             0.0)
        return model


cdef class IBM1(Model):
    """
    An IBM1 is a 0th-order model with lexical parameters only.
    """

    def __init__(self, LexicalParameters lex_parameters):
        self._lex_parameters = lex_parameters

    cpdef LexicalParameters lexical_parameters(self):
        return self._lex_parameters

    cpdef float likelihood(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i , f | e).

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: 1.0/(l + 1) * lex(f_j|e_i)
        """
        cdef float dist_parameter = 1.0 / len(e_snt)
        return dist_parameter * self._lex_parameters.get(e_snt[i], f_snt[j])

    cpdef float posterior(self, np.int_t[::1] e_snt, np.int_t[::1] f_snt, int i, int j):
        """
        Evaluate p(a_j = i | f, e) up to a normalisation constant.

        :param e_snt: e_0^l
        :param f_snt: f_1^l
        :param i: English position
        :param j: French position
        :return: lex(f_j|e_i)
        """
        return self._lex_parameters.get(e_snt[i], f_snt[j])

# TODO: write IBM2 (Vogel's parameterisation)