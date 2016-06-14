"""
:Authors: - Wilker Aziz
"""
import numpy as np
import lola.util as util
from lola.component import LexicalParameters, JumpParameters, BrownDistortionParameters
from lola.log_linear import LogLinearParameters
from lola.model import DefaultModel
from lola.feature_vector import FeatureMatrix
from lola.extractor import LexFeatures
from lola.corpus import Corpus
from collections import defaultdict
import logging


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


def read_extractor(e_corpus, f_corpus, args, line, i, extractors):
    try:
        cfg, [name, _] = util.re_sub('^([^:]+)(:)', '', line)
    except:
        raise ValueError('In line %d, expected extractor name: %s' % (i, line))

    if name in extractors:
        raise ValueError('Duplicate extractor name in line %d: %s', i, name)

    from lola.ff import WholeWordFeatures, AffixFeatures, CategoryFeatures
    cfg, extractor_type = util.re_key_value('type', cfg, optional=False, dtype=str)

    if extractor_type == 'WholeWordFeatures':
        extractors[name] = WholeWordFeatures.construct(e_corpus, f_corpus, cfg)
    elif extractor_type == 'AffixFeatures':
        extractors[name] = AffixFeatures.construct(e_corpus, f_corpus, cfg)
    elif extractor_type == 'CategoryFeatures':
        extractors[name] = CategoryFeatures.construct(e_corpus, f_corpus, cfg)
    else:
        raise ValueError('In line %d, got an unknown extractor type: %s' % (i, extractor_type))


def read_component(e_corpus, f_corpus, args, line, i, components, specs: defaultdict, extractors: list):

    try:
        cfg, [name, _] = util.re_sub('^([^:]+)(:)', '', line)
    except:
        raise ValueError('In line %d, expected component name: %s' % (i, line))

    if name in components:
        raise ValueError('Duplicate component name in line %d: %s', i, name)

    cfg, component_type = util.re_key_value('type', cfg, optional=False, dtype=str)
    cfg, model_number = util.re_key_value('model', cfg, optional=False)
    specs[model_number].append(name)

    if component_type == 'BrownLexical':
        components[name] = LexicalParameters(e_corpus.vocab_size(),
                                             f_corpus.vocab_size(),
                                             p=1.0 / f_corpus.vocab_size(),
                                             name=name)
    elif component_type == 'VogelJump':
        components[name] = JumpParameters(e_corpus.max_len(),
                                          f_corpus.max_len(),
                                          1.0 / (e_corpus.max_len() + f_corpus.max_len() + 1),
                                          name=name)
    elif component_type == 'BrownDistortion':
        components[name] = BrownDistortionParameters(e_corpus.max_len(),
                                                     1.0 / (e_corpus.max_len()),
                                                     name=name)
    elif component_type == 'LogLinearLexical':

        # get lexical feature extractors
        cfg, lex_extractors_names = util.re_key_value('extractors', cfg, optional=False)
        try:
            lex_extractors = [extractors[name] for name in lex_extractors_names]
        except KeyError:
            raise ValueError('In line %d, tried to use an undeclared extractor: %s' % (i, lex_extractors_names))

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
        components[name] = LogLinearParameters(e_corpus.vocab_size(),
                                               f_corpus.vocab_size(),
                                               weight_vector,
                                               feature_matrix,
                                               p=0.0,
                                               lbfgs_steps=sgd_steps,
                                               lbfgs_max_attempts=sgd_attempts,
                                               name=name)
    elif component_type == 'LogLinearSuffix':
        pass  # TODO
    elif component_type == 'LogLinearPrefix':
        pass  # TODO
    else:
        raise ValueError('Unkonwn component type in line %d: %s', i, component_type)


def parse_blocks(e_corpus: Corpus, f_corpus: Corpus, args, istream, first_line):
    components = {}
    extractors = {}
    specs = defaultdict(list)
    n_iterations = []
    block_type = None
    for i, line in enumerate(istream, first_line):
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            continue

        if line == '[components]':
            block_type = 'components'

        elif line == '[iterations]':
            block_type = 'iterations'

        elif line == '[extractors]':
            block_type = 'extractors'

        else:
            if block_type == 'components':
                read_component(e_corpus, f_corpus, args, line, i, components, specs, extractors)
            elif block_type == 'iterations':
                read_iteration(line, i, n_iterations)
            elif block_type == 'extractors':
                read_extractor(e_corpus, f_corpus, args, line, i, extractors)

    for name, extractor in extractors.items():
        print(name, extractor)

    return components, specs, n_iterations


def make_model(path, e_corpus: Corpus, f_corpus: Corpus, args) -> 'list[DefaultModel], list[int]':
    """

    :param path: path to configuration file
    :return:
    """

    with open(path) as fi:
        comps, comps_per_model, iterations_per_model = parse_blocks(e_corpus, f_corpus, args, fi, 1)
        selected = []
        models = []
        for m, iterations in enumerate(iterations_per_model, 1):
            names = comps_per_model[m]
            # add to the selection
            selected.extend([comps[name] for name in names])
            # create a model with the components selected for this round
            models.append(DefaultModel(selected))

        return models, iterations_per_model
