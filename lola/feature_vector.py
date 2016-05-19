import scipy.sparse as sparse
from lola.extractor import LexExampleFeatures
from lola.corpus import Corpus


class FeatureMatrix:
    """
    An initialized feature matrix, on which can be queried.
    """

    def __init__(self, e_corpus, f_corpus, features, number_of_features):
        """
        Initializes a feature matrix class
        :param e_corpus: an instance of Corpus (with NULL tokens)
        :param f_corpus: an instance of Corpus (without NULL tokens)
        :param features: a feature class that inherits the FeatureExtractor
        :param number_of_features: number of unique features
        """
        e_vocab_size = e_corpus.vocab_size()
        f_vocab_size = f_corpus.vocab_size()
        self._max_word_pairs = e_vocab_size * f_vocab_size
        self._max_d = number_of_features
        self._feature_vector = sparse.dok_matrix((self._max_word_pairs, self._max_d), dtype=bool)
        self._word_pair_dict = {}
        self._feature_dict = {}
        self.init_feature_vector(e_corpus, f_corpus, features)

    def init_feature_vector(self, e_corpus, f_corpus, features):
        """
        Initializes the feature matrix itself with the following parameters
        :param e_corpus: an instance of Corpus (with NULL tokens)
        :param f_corpus: an instance of Corpus (without NULL tokens)
        :param features: a feature class that inherits the FeatureExtractor
        :return: a sparse dok_matrix with word pairs x features
        """

        # set the indices for the dok_matrix
        word_pair_index = 0
        feature_index = 0

        # loop over all sentences in the corpus
        for s, (f_snt, e_snt) in enumerate(zip(f_corpus.itersentences(), e_corpus.itersentences()), 1):

            # loop over all words pairs in the sentence pairs
            for j in range(len(f_snt)):
                for i in range(len(e_snt)):

                    # add word pair, if not seen before, to dictionary
                    if (f_snt[j], e_snt[i]) not in self._word_pair_dict:
                        self._word_pair_dict[(f_snt[j], e_snt[i])] = word_pair_index
                        word_pair_index += 1

                    # extract all features and loop over them
                    features_list = features.extract(e_snt, f_snt, i, j)
                    for feature in features_list:

                        # add feature, if not seen before, to dictionary
                        if feature not in self._feature_dict.keys():
                            self._feature_dict[feature] = feature_index
                            feature_index += 1

                        # as all features in the feature list belong to the current word pair,
                        # set matrix position of word pair with feature to true
                        self.update_matrix(e_snt[i], f_snt[j], feature, True)

        return

    def update_matrix(self, e_word, f_word, feature, value):
        """
        Update the position in the feature matrix
        :param e_word: an integer representing an English word
        :param f_word: an integer representing a French word
        :param feature: a feature, represented by a string
        :param value: value with which the feature matrix should be updated
        :return:
        """
        word_pair_index = self.get_word_pair_index(f_word, e_word)
        feature_index = self.get_feature_index(feature)
        self._feature_vector[word_pair_index, feature_index] = value

    def max_word_pair_size(self):
        """

        :return: total amount of possible word-pairs (note, not actual amount of word pairs)
        """
        return self._max_word_pairs

    def max_feature_size(self):
        """

        :return: total amount of possible features (note, not actual amount of features)
        """
        return self._max_d

    def word_pair_size(self):
        """

        :return: amount of word pairs in the feature matrix
        """
        return len(self._word_pair_dict)

    def feature_size(self):
        """

        :return: amount of features in the feature matrix
        """
        return len(self._feature_dict)

    def get_feature_vector(self, f_word, e_word):
        """
        returns a feature vector of a word pair
        note that this function does not check on existence of word pair
        :param e_word: an integer representing an English word
        :param f_word: an integer representing a French word
        :return: feature vector of a word pair
        """
        word_pair_index = self.get_word_pair_index(f_word, e_word)
        return self._feature_vector[word_pair_index, :].todense()

    def get_feature_value(self, f_word, e_word, feature):
        """
        returns a feature value for a word pair and a feature
        note that this function does not check on existence of word pair
        :param e_word: an integer representing an English word
        :param f_word: an integer representing a French word
        :param feature: a feature, represented by a string
        :return: a bool with the value of the indexed word pair and feature
        """
        word_pair_index = self.get_word_pair_index(f_word, e_word)
        feature_index = self.get_feature_index(feature)
        return self._feature_vector[word_pair_index, feature_index]

    def get_word_pair_index(self, f_word, e_word):
        """
        returns the index of the word pair in the feature matrix
        :param e_word: an integer representing an English word
        :param f_word: an integer representing a French word
        :return: an integer representing the word pair in the feature matrix
        """
        return self._word_pair_dict[(f_word, e_word)]

    def get_feature_index(self, feature):
        """
        return the index of the feature in the feature matrix
        :param feature: a feature, represented by a string
        :return: an integer representing the feature in the feature matrix
        """
        return self._feature_dict[feature]

if __name__ == '__main__':

    F = Corpus('training/example.f')
    E = Corpus('training/example.e', null='<NULL>')

    lexFeatures = LexExampleFeatures(E, F)

    f = FeatureMatrix(E, F, lexFeatures, E.vocab_size() + F.vocab_size())
    print(f._feature_vector.todense())
