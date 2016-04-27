import numpy as np


def tokenize(path, bos=None):
    """
    This method tokenizes an input corpus and returns a stream of tokens.

    :param path: path to corpus
    :param bos: this is optional and if set it is added to the beginning of every sentence
    :return: an np.array of tokens, and an np.array of boundary positions
    """
    with open(path, 'r') as fi:
        tokens = []
        boundaries = []
        if bos:
            for line in fi.readlines():
                tokens.append(bos)
                tokens.extend(line.split())
                boundaries.append(len(tokens))
        else:
            for line in fi.readlines():
                tokens.extend(line.split())
                boundaries.append(len(tokens))
        return np.array(tokens, dtype='U'), np.array(boundaries, dtype=np.int)


class Corpus:
    """
    A corpus is a collection of sentences.
    Each sentence is a sequence of words.

    Internally, words are represented as integers for compactness and quick indexing using numpy arrays.

    Remark: This object offers no guarantee as to which exact index any word will get. Not even the NULL word.
    """

    def __init__(self, path, null=None):
        """
        Creates a corpus from a text file.
        The corpus is internally represented by a flat numpy array.

        :param path: path to text file (one sentence per line and no boundary markers)
        :param null: an optional NULL token to be added to the beginning of every sentence
        """

        with open(path, 'r') as fi:
            # read and tokenize the entire corpus
            # and if a null symbol is given, we place it at the beginning of the sentence
            # we also memorise the boundary positions
            tokens, self._boundaries = tokenize(path, bos=null)
            # use numpy to map tokens to integers
            # lookup converts from integers back to strings
            # inverse represents the corpus with words represented by integers
            self._lookup, self._inverse = np.unique(tokens, return_inverse=True)

    def itersentences(self):
        """Iterates over sentences"""
        a = 0
        for b in self._boundaries:
            yield self._inverse[a:b]  # this produces a view, not a copy ;)
            a = b

    def translate(self, i):
        """
        Translate an integer back to a string.
        :param i: index representing the word
        :return: original string
        """
        return self._lookup[i]

    def vocab_size(self):
        """Number of unique tokens (if the corpus was created with added NULL tokens, this will include it)"""
        return self._lookup.size

    def corpus_size(self):
        """Number of tokens in the corpus."""
        return self._inverse.size

    def n_sentences(self):
        """Number of sentences in the corpus."""
        return self._boundaries.size
