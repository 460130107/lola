"""
:Authors: - Wilker Aziz
"""

import numpy as np
import lola.util as util
from lola.component import LexicalParameters, UniformAlignment, JumpParameters, BrownDistortionParameters
from lola.log_linear import LogLinearParameters
from lola.feature_vector import FeatureMatrix
from lola.corpus import Corpus
from lola.model import DefaultModel
import logging


class ModelSpec:
    """
    Model specification e.g. components and number of iterations of EM.
    """

    def __init__(self, name: str, components: list, iterations: int):
        self.name = name
        self.components = tuple(components)
        self.iterations = iterations

    def __str__(self):
        return 'Model %s (iterations=%d): %s' % (self.name, self.iterations, ', '.join(self.components))

    def make(self, components) -> DefaultModel:
        if not all(name in components for name in self.components):
            raise ValueError('Missing components')
        return DefaultModel([components[name] for name in self.components])


class Config:

    def __init__(self):
        self._extractors = {}
        self._components = {}
        self._models = {}
        self._order = []

    def components(self):
        return self._components

    def has_extractor(self, name: str):
        return name in self._extractors

    def has_component(self, name: str):
        return name in self._components

    def has_model(self, name: str):
        return name in self._models

    def add_extractor(self, name: str, extractor):
        if not self.has_extractor(name):
            self._extractors[name] = extractor
        else:
            raise ValueError('Duplicate extractor name: %s' % name)

    def add_component(self, name: str, component):
        if not self.has_component(name):
            self._components[name] = component
        else:
            raise ValueError('Duplicate component name: %s' % name)

    def append_model(self, name: str, specs: ModelSpec):
        if not self.has_model(name):
            self._models[name] = specs
            self._order.append(name)
        else:
            raise ValueError('Duplicate model name: %s' % name)

    def itermodels(self) -> 'list[ModelSpec]':
        for name in self._order:
            yield self._models[name]

    def get_extractor(self, name: str):
        if name not in self._extractors:
            raise ValueError('Undeclared extractor: %s' % name)
        return self._extractors[name]

    def get_component(self, name: str):
        if name not in self._components:
            raise ValueError('Undeclared component: %s' % name)
        return self._components[name]


def dummy_action(e_corpus: Corpus, f_corpus: Corpus, args, line: str, i: int, state: Config):
    pass


def read_iteration(line, i, iterations):
    if line.startswith('#'):
        return
    line = line.strip()
    if not line:
        return
    try:
        iterations.append(int(line))
    except:
        raise ValueError('In line %d, expected number of iterations for model %d: %s' % (i, len(iterations) + 1, line))


def read_extractor(e_corpus: Corpus, f_corpus: Corpus, args, line: str, i: int, state: Config):
    try:
        cfg, [name, _] = util.re_sub('^([^:]+)(:)', '', line)
    except:
        raise ValueError('In line %d, expected extractor name: %s' % (i, line))

    if state.has_extractor(name):
        raise ValueError('Duplicate extractor name in line %d: %s', i, name)

    from lola.ff import WholeWordFeatures, AffixFeatures, CategoryFeatures
    cfg, extractor_type = util.re_key_value('type', cfg, optional=False, dtype=str)

    if extractor_type == 'WholeWordFeatures':
        state.add_extractor(name, WholeWordFeatures.construct(e_corpus, f_corpus, cfg))
    elif extractor_type == 'AffixFeatures':
        state.add_extractor(name, AffixFeatures.construct(e_corpus, f_corpus, cfg))
    elif extractor_type == 'CategoryFeatures':
        state.add_extractor(name, CategoryFeatures.construct(e_corpus, f_corpus, cfg))
    else:
        raise ValueError('In line %d, got an unknown extractor type: %s' % (i, extractor_type))


def read_component(e_corpus: Corpus, f_corpus: Corpus, args, line: str, i: int, state: Config):
    try:
        cfg, [name, _] = util.re_sub('^([^:]+)(:)', '', line)
    except:
        raise ValueError('In line %d, expected component name: %s' % (i, line))

    if state.has_component(name):
        raise ValueError('Duplicate component name in line %d: %s', i, name)

    cfg, component_type = util.re_key_value('type', cfg, optional=False, dtype=str)

    if component_type == 'BrownLexical':
        state.add_component(name, LexicalParameters(e_corpus.vocab_size(),
                                                    f_corpus.vocab_size(),
                                                    p=1.0 / f_corpus.vocab_size(),
                                                    name=name))
    elif component_type == 'UniformAlignment':
        state.add_component(name, UniformAlignment(name=name))
    elif component_type == 'VogelJump':
        state.add_component(name, JumpParameters(e_corpus.max_len(),
                                                 f_corpus.max_len(),
                                                 1.0 / (e_corpus.max_len() + f_corpus.max_len() + 1),
                                                 name=name))
    elif component_type == 'BrownDistortion':
        state.add_component(name, BrownDistortionParameters(e_corpus.max_len(),
                                                            1.0 / (e_corpus.max_len()),
                                                            name=name))
    elif component_type == 'LogLinearLexical':

        # get lexical feature extractors
        cfg, lex_extractors_names = util.re_key_value('extractors', cfg, optional=False)
        lex_extractors = []
        for extrator_name in lex_extractors_names:
            if state.has_extractor(extrator_name):
                lex_extractors.append(state.get_extractor(extrator_name))
            else:
                raise ValueError('In line %d, tried to use an undeclared extractor: %s' % (i, extrator_name))

        cfg, min_count = util.re_key_value('min-count', cfg, optional=True, default=1)
        cfg, max_count = util.re_key_value('max-count', cfg, optional=True, default=-1)

        # create a feature matrix based on feature extractors and configuration
        logging.info('Building feature matrix for %s (%s)', name, component_type)
        feature_matrix = FeatureMatrix(e_corpus, f_corpus, lex_extractors,
                                       min_occurences=min_count, max_occurrences=max_count)

        dimensionality = feature_matrix.get_feature_size()
        logging.info('Unique lexical features: %d', dimensionality)

        # create an initial parameter vector
        cfg, init_option = util.re_key_value('init', cfg, optional=True, default='normal')
        if init_option == 'uniform':
            weight_vector = np.full(dimensionality, 1.0 / dimensionality, dtype=np.float)
        else:  # random initialisation from a normal distribution with mean 0 and var 1.0
            weight_vector = np.random.normal(0, 1.0, dimensionality)

        # configure SGD
        cfg, sgd_steps = util.re_key_value('sgd-steps', cfg, optional=True, default=3)
        cfg, sgd_attempts = util.re_key_value('sgd-attempts', cfg, optional=True, default=5)

        # configure LogLinearParameters
        state.add_component(name, LogLinearParameters(e_corpus.vocab_size(),
                                                      f_corpus.vocab_size(),
                                                      weight_vector,
                                                      feature_matrix,
                                                      p=0.0,
                                                      lbfgs_steps=sgd_steps,
                                                      lbfgs_max_attempts=sgd_attempts,
                                                      name=name))
    elif component_type == 'LogLinearSuffix':
        pass  # TODO
    elif component_type == 'LogLinearPrefix':
        pass  # TODO
    else:
        raise ValueError('Unkonwn component type in line %d: %s', i, component_type)


def read_model(e_corpus: Corpus, f_corpus: Corpus, args, line: str, i: int, state: Config):
    try:
        cfg, [name, _] = util.re_sub('^([^:]+)(:)', '', line)
    except:
        raise ValueError('In line %d, expected model name: %s' % (i, line))

    if state.has_model(name):
        raise ValueError('Duplicate model name in line %d: %s', i, name)

    # get components
    cfg, components_names = util.re_key_value('components', cfg, optional=False)
    # sanity check
    for comp_name in components_names:
        if not state.has_component(comp_name):
            raise ValueError('In line %d, tried to use an undeclared component: %s' % (i, comp_name))
    # get number of iterations
    cfg, iterations = util.re_key_value('iterations', cfg, optional=False)

    # models are "appended" because the order matters
    state.append_model(name, ModelSpec(name, components_names, iterations))


def parse_blocks(e_corpus: Corpus, f_corpus: Corpus, args, istream, first_line):
    # types of blocks and their parsers
    header_to_action = {'[components]': read_component,
                        '[extractors]': read_extractor,
                        '[models]': read_model}
    config = Config()
    action = dummy_action

    for i, line in enumerate(istream, first_line):
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            continue

        if line in header_to_action:
            action = header_to_action[line]
        else:
            action(e_corpus, f_corpus, args, line, i, config)

    return config


def configure(path, e_corpus: Corpus, f_corpus: Corpus, args) -> Config:
    """
    :param path: path to configuration file
    :return: a Config object
    """

    with open(path) as fi:
        return parse_blocks(e_corpus, f_corpus, args, fi, 1)
