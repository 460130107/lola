from lola.event cimport Event, LexEvent, JumpEvent, DistEvent
from lola.event cimport LexContext, LexDecision
from lola.event cimport JumpContext, JumpDecision
from lola.event cimport DistContext, DistDecision
from libc.math cimport floor, fabs, exp
import re
from lola.util import re_key_value
cimport numpy as np


cdef class FeatureExtractor:

    def __init__(self, str name):
        self._name = name

    cpdef list extract(self, Event event, list features=[]):  # TODO: support real-valued features as well
        return features

    cpdef str name(self):
        return self._name


cdef class LexicalFeatureExtractor(FeatureExtractor):

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True,
                 str name='LexicalFeatureExtractor'):
        super(LexicalFeatureExtractor, self).__init__(name)
        self.e_corpus = e_corpus
        self.f_corpus = f_corpus
        self.extract_e = extract_e
        self.extract_f = extract_f
        self.extract_ef = extract_ef

    cpdef list extract(self, Event event, list features=[]):
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
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatureExtractor.parse_config(cfg)
        return LexicalFeatureExtractor(e_corpus, f_corpus, extract_e, extract_f, extract_ef)


cdef class WholeWordFeatureExtractor(LexicalFeatureExtractor):
    """
    Example class using the word itself as feature only (in both English as French)
    """

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True,
                 str name='WholeWordFeatureExtractor'):
        super(WholeWordFeatureExtractor, self).__init__(e_corpus, f_corpus, extract_e, extract_f, extract_ef, name)

    cpdef list extract(self, Event event, list features=[]):
        if not isinstance(event, LexEvent):
            raise ValueError('Expected LexEvent, got %s' % type(event))
        cdef:
            LexContext c = <LexContext>event.context
            LexDecision d = <LexDecision>event.decision
        e_str = self.e_corpus.translate(c.word())
        f_str = self.f_corpus.translate(d.word())
        if self.extract_e:
            features.append('e[i]=%s' % e_str)
        if self.extract_f:
            features.append('f[i]=%s' % f_str)
        if self.extract_ef:
            features.append('e[i]|f[i]=%s|%s' % (e_str, f_str))
        return features

    @staticmethod
    def parse_config(cfg):
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatureExtractor.parse_config(cfg)
        return cfg, [extract_e, extract_f, extract_ef]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, str cfg):
        cfg, [extract_e, extract_f, extract_ef] = WholeWordFeatureExtractor.parse_config(cfg)
        return WholeWordFeatureExtractor(e_corpus, f_corpus, extract_e, extract_f, extract_ef)


cdef class AffixFeatureExtractor(LexicalFeatureExtractor):

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True,
                 list suffix_sizes=[2,3,4], list prefix_sizes=[2,3,4],
                 size_t min_e_length=1, size_t min_f_length=1,
                 str name='AffixFeatureExtractor'):
        super(AffixFeatureExtractor, self).__init__(e_corpus, f_corpus, extract_e, extract_f, extract_ef, name)
        self.suffix_sizes = list(suffix_sizes)
        self.prefix_sizes = list(prefix_sizes)
        self.min_e_length = min_e_length
        self.min_f_length = min_f_length

    cpdef list extract(self, Event event, list features=[]):
        if not isinstance(event, LexEvent):
            raise ValueError('Expected LexEvent, got %s' % type(event))
        cdef:
            LexContext c = <LexContext>event.context
            LexDecision d = <LexDecision>event.decision
            size_t size
        e_str = self.e_corpus.translate(c.word())
        f_str = self.f_corpus.translate(d.word())

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
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatureExtractor.parse_config(cfg)
        cfg, suffix_sizes = re_key_value('suffix_sizes', cfg, optional=True, default=[])
        cfg, prefix_sizes = re_key_value('prefix_sizes', cfg, optional=True, default=[])
        cfg, min_e_length = re_key_value('min_e_length', cfg, optional=True, default=1)
        cfg, min_f_length = re_key_value('min_f_length', cfg, optional=True, default=1)
        return cfg, [extract_e, extract_f, extract_ef, suffix_sizes, prefix_sizes, min_e_length, min_f_length]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, str cfg):
        cfg, attrs = AffixFeatureExtractor.parse_config(cfg)
        return AffixFeatureExtractor(e_corpus, f_corpus, *attrs)


cdef class CategoryFeatureExtractor(LexicalFeatureExtractor):

    def __init__(self, Corpus e_corpus, Corpus f_corpus,
                 bint extract_e=True, bint extract_f=True, bint extract_ef=True,
                 str name='CategoryFeatureExtractor'):
        super(CategoryFeatureExtractor, self).__init__(e_corpus, f_corpus, extract_e, extract_f, extract_ef, name)
        self.digits_re = re.compile('\d')  # check for digits

    cpdef list extract(self, Event event, list features=[]):
        if not isinstance(event, LexEvent):
            raise ValueError('Expected LexEvent, got %s' % type(event))
        cdef:
            LexContext c = <LexContext>event.context
            LexDecision d = <LexDecision>event.decision

        e_str = self.e_corpus.translate(c.word())
        f_str = self.f_corpus.translate(d.word())
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
        cfg, [extract_e, extract_f, extract_ef] = LexicalFeatureExtractor.parse_config(cfg)
        return cfg, [extract_e, extract_f, extract_ef]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, str cfg):
        cfg, [extract_e, extract_f, extract_ef] = CategoryFeatureExtractor.parse_config(cfg)
        return CategoryFeatureExtractor(e_corpus, f_corpus, extract_e, extract_f, extract_ef)


cdef class JumpFeatureExtractor(FeatureExtractor):

    def __init__(self, str name='JumpFeatureExtractor', list bins=[]):
        super(JumpFeatureExtractor, self).__init__(name)
        self._bins = list(bins)

    cpdef list extract(self, Event event, list features=[]):
        if not isinstance(event, JumpEvent):
            raise ValueError('Expected a JumpEvent, got %s' % type(event))

        cdef JumpDecision d = <JumpDecision>event.decision
        features.append('jump=%d' % d.jump())
        cdef int abs_jump = <int>fabs(d.jump())
        cdef int bin_value
        for bin_value in self._bins:
            features.append('|jump|>%d=%s' % (bin_value, abs_jump > bin_value))

        return features

    @staticmethod
    def parse_config(cfg):
        """
        Convert a configuration string into values of relevant attributes for construction.
        :returns: cfg, [attributes]
        """
        cfg, bins = re_key_value('bins', cfg, optional=True, default=[])
        return cfg, [bins]

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, cfg):
        """
        Construct an instance of the extractor based on a configuration string (and a parallel corpus).
        """
        cfg, [bins] = JumpFeatureExtractor.parse_config(cfg)
        return JumpFeatureExtractor(bins=bins)


cdef class DistortionFeatureExtractor(FeatureExtractor):

    def __init__(self, str name='DistortionFeatureExtractor'):
        super(DistortionFeatureExtractor, self).__init__(name)

    cpdef list extract(self, Event event, list features=[]):
        if not isinstance(event, DistEvent):
            raise ValueError('Expected a DistEvent, got %s' % type(event))

        cdef:
            DistContext c = <DistContext>event.context
            DistDecision d = <DistDecision>event.decision
        features.append('i=%d' % d.i())
        features.append('j=%d' % c.j)
        features.append('l=%d' % c.l)
        features.append('m=%d' % c.m)
        cdef:
            int jump = d.i() - <int>floor(float(c.j * c.l) / c.m)
        features.append('jump=%d' % jump)
        features.append('|jump|=%d' % <int>fabs(jump))
        if jump == 0:
            features.append('direction=0')
        elif jump > 0:
            features.append('direction=+')
        else:
            features.append('direction=-')

        return features

    @staticmethod
    def parse_config(cfg):
        """
        Convert a configuration string into values of relevant attributes for construction.
        :returns: cfg, [attributes]
        """
        return cfg, []

    @staticmethod
    def construct(Corpus e_corpus, Corpus f_corpus, cfg):
        """
        Construct an instance of the extractor based on a configuration string (and a parallel corpus).
        """
        return DistortionFeatureExtractor()

