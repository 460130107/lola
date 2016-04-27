import numpy as np
import itertools


def tokenize(path, bos=None, eos=''):
    """
    This method tokenizes an input corpus and returns a single stream of tokens.
    It requires an end-of-sentence marker, otherwise it would not be possible to recover sentence boundaries.
    It may use a begin-of-sentence marker as well.
    These markers should be unique strings that do not conflict with real vocabulary entries.

    :param path: path to corpus
    :param bos: this is optional and if set it is added to the beginning of every sentence
    :param eos: this is not optional, it must be a unique string to represent the boundary
        it may be an empty string, but not None
    :return: a list of tokens
    """
    if eos is None:
        raise ValueError('I need an EOS string: it can be emtpy, but it cannot be None')
    with open(path, 'r') as fi:
        corpus = []
        if bos:
            for line in fi.readlines():
                corpus.append(bos)
                corpus.extend(line.split())
                corpus.append(eos)
        else:
            for line in fi.readlines():
                corpus.extend(line.split())
                corpus.append(eos)
        return corpus


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

        self._null_added = null is not None

        with open(path, 'r') as fi:
            # read and tokenize the entire corpus
            # we mark sentence boundary with empty tokens eos=''
            # and if a null symbol is given, we place it at the beginning of the sentence
            text_corpus = np.array(tokenize(path, bos=null, eos=''), dtype='U')
            # memorise the boundary positions
            self._boundaries = np.where(text_corpus == '')[0]
            # use numpy to map tokens to integers
            # lookup converts from integers back to strings
            # inverse represents the corpus with words represented by integers
            self._lookup, self._inverse = np.unique(text_corpus, return_inverse=True)

    def itersentences(self):
        """Iterates over sentences"""
        a = 0
        for b in self._boundaries:
            yield self._inverse[a:b]
            a = b + 1
        # this makes it robust to having or not having a boundary symbol in the last sentence
        if a < self._inverse.size:
            yield self._inverse[a:]

    def translate(self, i):
        """
        Translate an integer back to a string.
        :param i: index representing the word
        :return: original string
        """
        return self._lookup[i]

    def vocab_size(self):
        """Number of unique tokens (if the corpus was created with added NULL tokens, this will include it)"""
        return self._lookup.size - 1  # we must discount the boundary symbol
