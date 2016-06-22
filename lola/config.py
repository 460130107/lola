"""
:Authors: - Wilker Aziz
"""

import numpy as np
import lola.util as util
from lola.component import LexicalParameters, UniformAlignment, JumpParameters, BrownDistortionParameters
from lola.llcomp import LogLinearComponent
from lola.event import LexEventSpace
from lola.event import JumpEventSpace
from lola.event import DistEventSpace
from lola.fmatrix import make_dense_matrices, make_sparse_matrices
from lola.corpus import Corpus
from lola.model import DefaultModel
import logging
from lola.ff import LexicalFeatureExtractor, JumpFeatureExtractor, DistortionFeatureExtractor
import sys


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


def make_loglinear_component(e_corpus: Corpus, f_corpus: Corpus, component_type: str, name: str, cfg: str, state: Config, line_number: int):
    if component_type == 'LogLinearLexical':
        event_space = LexEventSpace(e_corpus.vocab_size(), f_corpus.vocab_size())
        supported_extractor = LexicalFeatureExtractor
    elif component_type == 'LogLinearJump':
        event_space = JumpEventSpace(e_corpus.max_len())
        supported_extractor = JumpFeatureExtractor
    elif component_type == 'LogLinearDistortion':
        event_space = DistEventSpace(e_corpus.max_len())
        supported_extractor = DistortionFeatureExtractor
    else:
        raise ValueError('Unknown component type: %s' % component_type)

    # get lexical feature extractors
    cfg, extractor_names = util.re_key_value('extractors', cfg, optional=False)
    extractors = []
    for extrator_name in extractor_names:
        if state.has_extractor(extrator_name):
            extractor = state.get_extractor(extrator_name)
            if not isinstance(extractor, supported_extractor):
                raise ValueError('%s component only takes extractors of type %s, got %s' % (component_type, supported_extractor, type(extractor)))
            extractors.append(extractor)
        else:
            raise ValueError('In line %d, tried to use an undeclared extractor: %s' % (line_number, extrator_name))

    cfg, min_count = util.re_key_value('min-count', cfg, optional=True, default=1)
    cfg, max_count = util.re_key_value('max-count', cfg, optional=True, default=-1)

    # create a feature matrix based on feature extractors and configuration
    logging.info('Building dense feature matrix for %s (%s)', name, component_type)
    dense_matrix = make_dense_matrices(event_space, e_corpus, f_corpus, extractors)
    logging.info('Unique features (%s): dense=%d', name, dense_matrix.dimensionality())
    logging.info('Building sparse feature matrix for %s (%s)', name, component_type)
    sparse_matrix = make_sparse_matrices(event_space, e_corpus, f_corpus, extractors,
                                           min_occurrences={}, max_occurrences={})
    logging.info('Unique features (%s): sparse=%d', name,
                 sparse_matrix.dimensionality())

    #print('MATRIX')
    #dense_matrix.pp(e_corpus, f_corpus, sys.stdout)

    dimensionality = dense_matrix.dimensionality() + sparse_matrix.dimensionality()

    # create an initial parameter vector
    cfg, init_option = util.re_key_value('init', cfg, optional=True, default='normal')
    if init_option == 'uniform':
        weight_vector = np.full(dimensionality, 1.0 / dimensionality, dtype=np.float)
    else:  # random initialisation from a normal distribution with mean 0 and var 1.0
        weight_vector = np.random.normal(0, 1.0, dimensionality)

    # configure SGD
    cfg, sgd_steps = util.re_key_value('sgd-steps', cfg, optional=True, default=3)
    cfg, sgd_attempts = util.re_key_value('sgd-attempts', cfg, optional=True, default=5)
    cfg, regulariser_strength = util.re_key_value('regulariser-strength', cfg, optional=True, default=0.0)
    logging.debug('Optimisation settings (%s): sgd-steps=%d sgd-attempts=%d regulariser-strength=%f',
                  name, sgd_steps, sgd_attempts, regulariser_strength)

    # configure LogLinearParameters
    state.add_component(name, LogLinearComponent(weight_vector[:dense_matrix.dimensionality()],  # dense
                                                 weight_vector[dense_matrix.dimensionality():],  # sparse
                                                 dense_matrix,
                                                 sparse_matrix,
                                                 event_space,
                                                 lbfgs_steps=sgd_steps,
                                                 lbfgs_max_attempts=sgd_attempts,
                                                 regulariser_strength=regulariser_strength,
                                                 name=name))


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

    # we import the known implementations here
    from lola.ff import IBM1Probabilities, WholeWordFeatureExtractor, AffixFeatureExtractor
    from lola.ff import CategoryFeatureExtractor, LengthFeatures
    from lola.ff import JumpFeatureExtractor
    cfg, extractor_type = util.re_key_value('type', cfg, optional=False, dtype=str)

    try:
        implementation = eval(extractor_type)
        state.add_extractor(name, implementation.construct(e_corpus, f_corpus, cfg))
    except NameError:
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
                                                 1.0 / (2 * e_corpus.max_len() + 1),
                                                 name=name))
    elif component_type == 'BrownDistortion':
        state.add_component(name, BrownDistortionParameters(e_corpus.max_len(),
                                                            1.0 / (e_corpus.max_len()),
                                                            name=name))
    else:
        make_loglinear_component(e_corpus, f_corpus, component_type, name, cfg, state, i)


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
