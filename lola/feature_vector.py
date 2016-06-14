import scipy.sparse as sparse
import numpy as np
from lola.extractor import LexFeatures
from lola.corpus import Corpus
from collections import defaultdict
from lola.ff import extract_lexical_features
from lola.ff import LexicalFeatures


class FeatureMatrix:
    """
    An initialized feature matrix, on which can be queried.
    """

    def __init__(self, e_corpus, f_corpus, extractors: 'list[LexicalFeatures]', min_occurences=1, max_occurrences=-1):
        """
        Initializes a feature matrix class
        :param e_corpus: an instance of Corpus (with NULL tokens)
        :param f_corpus: an instance of Corpus (without NULL tokens)
        :param extractors: a list of extractors of type LexicalFeature
        :param number_of_features: number of unique features
        """
        self.e_vocab_size = e_corpus.vocab_size()
        self.f_vocab_size = f_corpus.vocab_size()
        self._min_occurrences = min_occurences
        self._max_occurrences = max_occurrences
        # our feature dictionary stores pairs: (feature id, feature count in the whole corpus)
        # we use a defaultdict so that pairs are automatically created for us
        # we mark a newly created feature with a negative id id and a 0 count
        self._feature_dict = defaultdict(lambda: [-1, 0])
        self._id_to_str = []
        self._feature_vector, self._max_rows, self._max_cols = self.initialise(e_corpus, f_corpus, extractors)

    def initialise(self, e_corpus, f_corpus, extractors: 'list[LexicalFeatures]'):
        """
        Initializes the feature matrix itself with the following parameters
        :param e_corpus: an instance of Corpus (with NULL tokens)
        :param f_corpus: an instance of Corpus (without NULL tokens)
        :param extractor: a feature class that inherits the FeatureExtractor
        :return: a sparse csr_matrix with word pairs x features
        """

        # set the indices for the feature space
        word_pair_features = [defaultdict(list) for _ in range(f_corpus.vocab_size())]

        # loop over all sentences in the corpus
        for s, (f_snt, e_snt) in enumerate(zip(f_corpus.itersentences(), e_corpus.itersentences()), 1):

            # loop over all words pairs in the sentence pairs
            for j in range(len(f_snt)):
                f_features = word_pair_features[f_snt[j]]

                for i in range(len(e_snt)):
                    e_features = f_features[e_snt[i]]

                    if len(e_features) != 0:
                        continue  # we have nothing to do in this case, because this word pair has already been seen

                    for feature in extract_lexical_features(e_snt[i], f_snt[j], extractors):
                        # we try to retrieve information about the feature
                        # namely, a tuple containing its id and its count
                        # the count information concerns the whole corpus and is used in order to prune rare features
                        f_info = self._feature_dict[feature]
                        if f_info[0] == -1:  # we haven't yet seen this feature
                            # thus we update its id
                            f_info[0] = len(self._feature_dict) - 1
                            self._id_to_str.append(feature)
                        # and we increment the feature count
                        f_info[1] += 1
                        e_features.append(f_info)  # fire this feature for the given (e,f) pair

        # give it a chance to prune the space of features
        n_deleted_features = 0
        if self._min_occurrences > 1 or self._max_occurrences > 0:
            # here we clean up the feature space
            self._id_to_str = []
            for f_str, f_info in sorted(self._feature_dict.items(), key=lambda kv: kv[1][0]):
                # we may want to prune this feature
                if f_info[1] < self._min_occurrences or (0 < self._max_occurrences < f_info[1]):
                    f_info[0] = -1  # first we invalidate its id
                    n_deleted_features += 1  # then we increment the number of deleted features
                else:  # if we are not pruning, we might be shifting its id taking deleted features into account
                    f_info[0] -= n_deleted_features
                    self._id_to_str.append(f_str)

        r = e_corpus.vocab_size()  # max rows
        d = len(self._feature_dict) - n_deleted_features  # max columns (we discard deleted features)
        # now we can construct csr_matrix objects
        # for each F word we have one csr_matrix
        for f in range(f_corpus.vocab_size()):
            word_pair_features[f] = self.convert_to_csr(word_pair_features[f], max_rows=r, max_columns=d)
        # when we get here, we will have converted all (python) dictionary of features to (scipy) csr_matrix objects
        # now we just convert this big list to a numpy army
        return np.array(word_pair_features), r, d

    def sparse_zero_vec(self) -> sparse.csr_matrix:
        return sparse.csr_matrix((1, self._max_cols), dtype=float)

    def dense_zero_vec(self) -> np.array:
        return np.zeros(self._max_cols, dtype=float)

    @staticmethod
    def convert_to_csr(feature_dict, max_rows, max_columns) -> sparse.csr_matrix:
        """

        :param feature_dict: dictionary with English words as key and the feature (integer) as value
        :param max_rows: number of rows
        :param max_columns: number of columns
        :return: csr_matrix with counts
        """
        # dok_matrix are ideal for constructing sparse matrices
        dok = sparse.dok_matrix((max_rows, max_columns), dtype=int)
        for row, columns in feature_dict.items():
            # skip deleted features (those whose id are negative)
            for column, global_count in filter(lambda info: info[0] >= 0, columns):
                column = column
                dok[row, column] += 1  # count instead of boolean
        # csr_matrix are ideal for operations such as addition and dot product
        # thus after the sparse matrix is created we convert it to a csr_matrix
        return dok.tocsr()


    def get_feature_vector(self, f: int, e: int) -> sparse.csr_matrix:
        """
        returns a feature vector of a word pair
        note that this function does not check on existence of word pair
        :param f: an integer representing a French word
        :param e: an integer representing an English word
        :return: feature vector of a word pair
        """
        return self._feature_vector[f][e]

    def get_feature_value(self, f: int, e: int, feature: int):
        """
        returns a feature value for a word pair and a feature
        note that this function does not check on existence of word pair
        :param f: an integer representing a French word
        :param e: an integer representing an English word
        :param feature: a feature, represented by a string
        :return: a bool with the value of the indexed word pair and feature
        """
        feature_index = self.get_feature_index(feature)
        return self._feature_vector[f][e, feature_index]

    def get_feature_index(self, feature):
        """
        return the index of the feature in the feature matrix
        :param feature: a feature, represented by a string
        :return: an integer representing the feature in the feature matrix
        """
        return self._feature_dict[feature][0]  # index is in the first position

    def get_shape(self):
        return self._max_rows, self._max_cols

    def get_feature_size(self):
        return self._max_cols

    def get_feature_string(self, fid: int) -> str:
        return self._id_to_str[fid]

if __name__ == '__main__':

    F = Corpus('training/example.f')
    E = Corpus('training/example.e', null='<NULL>')

    lexFeatures = LexFeatures(E, F)

    f = FeatureMatrix(E, F, lexFeatures)
    print(f._feature_vector[0].todense())
