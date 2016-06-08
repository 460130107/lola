import scipy.sparse as sparse
import numpy as np
from lola.extractor import LexExampleFeatures
from lola.corpus import Corpus
from collections import defaultdict


class FeatureMatrix:
    """
    An initialized feature matrix, on which can be queried.
    """

    def __init__(self, e_corpus, f_corpus, features):
        """
        Initializes a feature matrix class
        :param e_corpus: an instance of Corpus (with NULL tokens)
        :param f_corpus: an instance of Corpus (without NULL tokens)
        :param features: a feature class that inherits the FeatureExtractor
        :param number_of_features: number of unique features
        """
        self.e_vocab_size = e_corpus.vocab_size()
        self.f_vocab_size = f_corpus.vocab_size()
        self._feature_dict = {}
        self._feature_vector, self._max_rows, self._max_cols = self.init_feature_vector(e_corpus, f_corpus, features)

    def init_feature_vector(self, e_corpus, f_corpus, extractor):
        """
        Initializes the feature matrix itself with the following parameters
        :param e_corpus: an instance of Corpus (with NULL tokens)
        :param f_corpus: an instance of Corpus (without NULL tokens)
        :param extractor: a feature class that inherits the FeatureExtractor
        :return: a sparse dok_matrix with word pairs x features
        """

        # set the indices for the dok_matrix
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
                    for feature in extractor.extract(e_snt, f_snt, i, j):
                        f_id = self._feature_dict.get(feature, None)  # try to get the feature id
                        if f_id is None:  # if there isn't
                            f_id = len(self._feature_dict)  # we get the next available id (starting from 0)
                            self._feature_dict[feature] = f_id

                        e_features.append(f_id)

        r = e_corpus.vocab_size()  # max rows
        d = len(self._feature_dict)  # max columns
        # now we can construct dok_matrix objects
        # for each F word we have one dok_matrix
        for f in range(f_corpus.vocab_size()):
            word_pair_features[f] = self.convert_to_dok(word_pair_features[f], max_rows=r, max_columns=d)
        # when we get here, we will have converted all (python) dictionary of features to (scipy) dok_matrix objects
        # now we just convert this big list to a numpy army
        return np.array(word_pair_features), r, d

    def zero_vec(self):
        return sparse.dok_matrix((1, self._max_cols))

    @staticmethod
    def convert_to_dok(feature_dict, max_rows, max_columns):
        """

        :param feature_dict: dictionary with English words as key and the feature (integer) as value
        :param max_rows: amount of rows
        :param max_columns: amount of columns
        :return: dok_matrix with boolean values
        """
        matrix = sparse.dok_matrix((max_rows, max_columns), dtype=bool)
        for row, columns in feature_dict.items():
            for column in columns:
                matrix[row, column] = True
        return matrix

    def get_feature_vector(self, f, e):
        """
        returns a feature vector of a word pair
        note that this function does not check on existence of word pair
        :param f: an integer representing a French word
        :param e: an integer representing an English word
        :return: feature vector of a word pair
        """
        return self._feature_vector[f][e, :]

    def get_feature_value(self, f, e, feature):
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
        return self._feature_dict[feature]

    def get_feature_size(self):
        return len(self._feature_dict)

if __name__ == '__main__':

    F = Corpus('training/example.f')
    E = Corpus('training/example.e', null='<NULL>')

    lexFeatures = LexExampleFeatures(E, F)

    f = FeatureMatrix(E, F, lexFeatures)
    print(f._feature_vector[0].todense())
