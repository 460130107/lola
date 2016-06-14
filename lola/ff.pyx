from lola.corpus cimport Corpus
import re
from lola.util import re_key_value


cdef class LexicalFeatures:

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True):
        self.e_corpus = e_corpus
        self.f_corpus = f_corpus
        self.extract_e = extract_e
        self.extract_f = extract_f
        self.extract_ef = extract_ef

    cpdef list extract(self, int e, int f, list features=[]):
        """
        :param e: English word (id)
        :param f: French word (id)
        :param features: list of active features
        :returns: list of active features
        """
        return features

    @staticmethod
    def parse_config(cfg):
        """
        Convert a configuration string into values of relevant attributes for construction.
        :returns: cfg, [attributes]
        """
        cfg, extract_e = re_key_value('extract_e', cfg, optional=True, default=True)
        cfg, extract_f = re_key_value('extract_f', cfg, optional=True, default=True)
        cfg, extract_ef = re_key_value('extract_ef', cfg, optional=True, default=True)
        return cfg, [extract_e, extract_f, extract_ef]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, cfg):
        """
        Construct an instance of the extractor based on a configuration string (and a parallel corpus).
        """
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatures.parse_config(cfg)
        return LexicalFeatures(e_corpus, f_corpus, extract_e, extract_f, extract_ef)


cpdef list extract_lexical_features(int e, int f, list lexical_extractors):
    """
    :param e: English word (id)
    :param f: French word (id)
    :param lexical_extractors: list of feature extractors of type LexicalFeatures
    :returns: list of active features
    """
    cdef:
        LexicalFeatures extractor
        list features = []

    for extractor in lexical_extractors:
        extractor.extract(e, f, features)

    return features


cdef class WholeWordFeatures(LexicalFeatures):
    """
    Example class using the word itself as feature only (in both English as French)
    """

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True):
        super(WholeWordFeatures, self).__init__(e_corpus, f_corpus, extract_e, extract_f, extract_ef)

    cpdef list extract(self, int e, int f, list features=[]):
        e_str = self.e_corpus.translate(e)
        f_str = self.f_corpus.translate(f)
        if self.extract_e:
            features.append('e[i]=%s' % e_str)
        if self.extract_f:
            features.append('f[i]=%s' % f_str)
        if self.extract_ef:
            features.append('e[i]|f[i]=%s|%s' % (e_str, f_str))
        return features

    @staticmethod
    def parse_config(cfg):
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatures.parse_config(cfg)
        return cfg, [extract_e, extract_f, extract_ef]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, str cfg):
        cfg, [extract_e, extract_f, extract_ef] = WholeWordFeatures.parse_config(cfg)
        return WholeWordFeatures(e_corpus, f_corpus, extract_e, extract_f, extract_ef)


cdef class AffixFeatures(LexicalFeatures):

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True,
                 list suffix_sizes=[2,3,4], list prefix_sizes=[2,3,4],
                 size_t min_e_length=1, size_t min_f_length=1):
        super(AffixFeatures, self).__init__(e_corpus, f_corpus, extract_e, extract_f, extract_ef)
        self.suffix_sizes = list(suffix_sizes)
        self.prefix_sizes = list(prefix_sizes)
        self.min_e_length = min_e_length
        self.min_f_length = min_f_length

    cpdef list extract(self, int e, int f, list features=[]):
        cdef size_t size
        e_str = self.e_corpus.translate(e)
        f_str = self.f_corpus.translate(f)

        if self.extract_e and len(e_str) >= self.min_e_length:
            for size in self.suffix_sizes:
                features.append('e[j][-%d:]=%s' % (size, e_str[-size:]))
            for size in self.prefix_sizes:
                features.append('e[j][:%d]=%s' % (size, e_str[:size]))

        if self.extract_f and len(f_str) >= self.min_f_length:
            for size in self.suffix_sizes:
                features.append('f[j][-%d:]=%s' % (size, f_str[-size:]))
            for size in self.prefix_sizes:
                features.append('f[j][:%d]=%s' % (size, f_str[:size]))

        if self.extract_ef and len(e_str) >= self.min_e_length and len(f_str) >= self.min_f_length:
            for size in self.suffix_sizes:
                features.append('e[j][-%d:]|f[j][-%d:]=%s|%s' % (size, size, e_str[-size:], f_str[-size:]))
            for size in self.prefix_sizes:
                features.append('e[j][:%d]|f[j][:%d]=%s|%s' % (size, size, e_str[:size], f_str[:size]))

        return features

    @staticmethod
    def parse_config(cfg):
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatures.parse_config(cfg)
        cfg, suffix_sizes = re_key_value('suffix_sizes', cfg, optional=True, default=[])
        cfg, prefix_sizes = re_key_value('prefix_sizes', cfg, optional=True, default=[])
        cfg, min_e_length = re_key_value('min_e_length', cfg, optional=True, default=1)
        cfg, min_f_length = re_key_value('min_f_length', cfg, optional=True, default=1)
        return cfg, [extract_e, extract_f, extract_ef, suffix_sizes, prefix_sizes, min_e_length, min_f_length]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, str cfg):
        cfg, attrs = AffixFeatures.parse_config(cfg)
        return AffixFeatures(e_corpus, f_corpus, *attrs)


cdef class CategoryFeatures(LexicalFeatures):

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True):
        super(CategoryFeatures, self).__init__(e_corpus, f_corpus, extract_e, extract_f, extract_ef)
        self.digits_re = re.compile('\d')  # check for digits

    cpdef list extract(self, int e, int f, list features=[]):
        e_str = self.e_corpus.translate(e)
        f_str = self.f_corpus.translate(f)
        if self.extract_e:
            features.append('E-has-digits=%s' % bool(self.digits_re.search(e_str)))
        if self.extract_f:
            features.append('F-has-digits=%s' % bool(self.digits_re.search(f_str)))
        if self.extract_ef:
            features.append('E-has-digits|F-has-digits=%s|%s' %
                            (bool(self.digits_re.search(e_str)), bool(self.digits_re.search(f_str))))
        return features

    @staticmethod
    def parse_config(cfg):
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatures.parse_config(cfg)
        return cfg, [extract_e, extract_f, extract_ef]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, str cfg):
        cfg, [extract_e, extract_f, extract_ef] = CategoryFeatures.parse_config(cfg)
        return CategoryFeatures(e_corpus, f_corpus, extract_e, extract_f, extract_ef)
