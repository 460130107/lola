"""
A module with efficient feature representations.

"""

from lola.ff cimport extract_lexical_features

from scipy import sparse
from collections import defaultdict

import numpy as np
cimport numpy as np
cimport cython


cdef class LexicalFeatureMatrix:
    """
    This object holds feature matrices describing each generating context (English words).
    Each feature matrix describes decisions (French words) in terms of sparse features.
    """

    def __init__(self, list matrices, list reversed_index, size_t d):
        self._matrices = matrices
        self._reversed_index = reversed_index
        self._d = d

    cpdef object sparse_zero_vec(self):
        return sparse.csr_matrix((1, self._d), dtype=float)

    cpdef np.float_t[::1] dense_zero_vec(self):
        return np.zeros(self._d, dtype=float)

    cpdef size_t dimensionality(self):
        return self._d

    cpdef object feature_matrix(self, int e):
        """
        Return the feature matrix associated with a context (English word).

        :param e: context
        :return: csr_matrix (rows are French words, columns are features)
        """
        return self._matrices[e]

    cpdef object feature_vector(self, int e, int f):
        """
        Return the feature vector associated with an event where
            the context is the English word
            and the decision is the French word
        The feature vector is represented as a compressed sparse row,
            thus an instance of csr_matrix whose shape is (1, dimensionality).

        :param e: context
        :param f: decision
        :return: csr_matrix (1 row, columns are features)
        """
        return self._matrices[e][f]

    cpdef object raw_feature_value(self, size_t column):
        cdef Feature feature = self._reversed_index[column]
        return feature.value


cdef class Feature:
    """
    This object simply represents a feature, it has an id, a global count, and a value.
    """

    def __init__(self, int id=-1, int count=0, object value=None, str parent=''):
        self.id = id
        self.count = count
        self.value = value
        self.parent = parent

    def __str__(self):
        return '{0}::{1}'.format(self.parent, self.value)


cdef object convert_to_csr(feature_dict, size_t max_rows, size_t max_columns):
    """
    Convert a python dictionary mapping French words to Feature objects into a csr_matrix object.

    :param feature_dict: dictionary with French words (ids) as key and the feature (integer) as value
    :param max_rows: number of rows
    :param max_columns: number of columns
    :return: csr_matrix with counts
    """
    cdef:
        int f
        list features
        Feature feature
    # dok_matrix are good for constructing sparse matrices
    dok = sparse.dok_matrix((max_rows, max_columns), dtype=int)
    for f, features in feature_dict.items():
        for feature in features:
            if feature.id < 0:  # skip deleted features
                continue
            dok[f, feature.id] += 1
    return dok.tocsr()


cpdef LexicalFeatureMatrix make_lexical_matrices(Corpus e_corpus,
                                                 Corpus f_corpus,
                                                 extractors,
                                                 int min_occurrences=1,
                                                 int max_occurrences=-1):
    """
    Initializes the feature matrix itself with the following parameters
    :param e_corpus: an instance of Corpus (with NULL tokens)
    :param f_corpus: an instance of Corpus (without NULL tokens)
    :param extractors: a collection of LexicalFeatures extractors
    :return: a sparse csr_matrix with word pairs x features
    """

    cdef:
        size_t S = e_corpus.n_sentences()
        size_t n_features = 0
        size_t s
        np.int_t[::1] e_snt, f_snt
        size_t i, j
        int e_i, f_j
        Feature feature
        list reverse_feature_index = []
        list word_pair_features = [defaultdict(list) for _ in range(e_corpus.vocab_size())]
        object feature_repo = defaultdict(Feature)  # repository of features

    # Loop over all sentence pairs gathering features for word pairs
    for s in range(S):
        e_snt = e_corpus.sentence(s)
        f_snt = f_corpus.sentence(s)

        # Loop over all words pairs in the sentence pairs
        for i in range(e_snt.shape[0]):
            e_i = e_snt[i]
            features_for_e = word_pair_features[e_i]  # this is a dictionary

            for j in range(f_snt.shape[0]):
                f_j = f_snt[j]
                features_for_e_and_f = features_for_e[f_j]  # this is a list

                if len(features_for_e_and_f) != 0:
                    continue  # we have nothing to do in this case, because this word pair has already been seen

                for raw_feature_value in extract_lexical_features(e_i, f_j, extractors):
                    # we try to retrieve information about the feature
                    # namely, a tuple containing its id and its count
                    # the count information concerns the whole corpus and is used in order to prune rare features
                    feature = feature_repo[raw_feature_value]
                    if feature.id == -1:  # we haven't yet seen this feature
                        # thus we update its id and value
                        feature.id = n_features
                        feature.value = raw_feature_value
                        # update the total number of unique features
                        n_features += 1
                        # and maintain a reverse index
                        reverse_feature_index.append(feature)
                    # and we increment that feature's count
                    feature.count += 1
                    # finally, we fire this feature for the given (e,f) pair
                    features_for_e_and_f.append(feature)

    # Give it a chance to prune the space of features
    cdef:
        size_t n_deleted_features = 0
        list selected_features = []
    if min_occurrences > 1 or max_occurrences > 0:
        # here we clean up the feature space
        selected_features = []
        for feature in reverse_feature_index:
            if feature.count < min_occurrences or (0 < max_occurrences < feature.count):
                feature.id = -1  # first we invalidate its id
                n_deleted_features += 1  # then we increment the number of deleted features
            else:  # if we are not pruning, we might be shifting its id taking deleted features into account
                feature.id -= n_deleted_features
                selected_features.append(feature)
        # update the reverse index and the total number of active features
        reverse_feature_index = selected_features
        n_features -= n_deleted_features

    # Build sparse matrices
    cdef:
        size_t r = f_corpus.vocab_size()  # max rows
        size_t d = n_features
        int e, f
    # now we can construct csr_matrix objects
    # for each English word e we have one csr_matrix where
    # each row represents a French word f and each column represents a feature phi relating e and f
    for e in range(e_corpus.vocab_size()):
        word_pair_features[e] = convert_to_csr(word_pair_features[e], r, d)
    # when we get here, we will have converted all (python) dictionary of features to (scipy) csr_matrix objects
    return LexicalFeatureMatrix(word_pair_features, reverse_feature_index, d)


