"""
:Authors: - Wilker Aziz
"""
import numpy as np


def uniform_lexical(e_vocab_size, f_vocab_size, value=None):
    """
    This returns a collection of |V_E| categorical distributions,
        each of which defined over the entire vocabulary of French words.
    The distribution is initialised uniformly unless a certain value is given.

    :param e_vocab_size: size of English vocabulary, i.e. |V_E|
    :param f_vocab_size: size of French vocabulary, i.e. |V_F|
    :param value: constant value for lex(e|f) which defaults to 1/|V_F|
    :return: |V_E| x |V_F| numpy array
    """
    # this is the uniform probability lex(f|e) = 1.0 / |V_F|
    if value is None:
        value = 1.0/f_vocab_size
    # we create |V_E| (uniform) categorical distributions
    # each of which has a support of |V_F| French words
    return np.full((e_vocab_size, f_vocab_size), value, dtype=np.float)


def counts_lexical(e_vocab_size, f_vocab_size):
    """
    This returns a 0-initialised matrix of partial counts.
    It is just wraper around uniform_lexical.
    """
    return uniform_lexical(e_vocab_size, f_vocab_size, 0.0)
