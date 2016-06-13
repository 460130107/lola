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


def read_component(e_corpus, f_corpus, args, line, i, components, specs: defaultdict):

    try:
        cfg, [name] = util.re_sub('^([^:]+)', '', line)
    except:
        raise ValueError('In line %d, expected component name: %s' % (i, line))

    if name in components:
        raise ValueError('Duplicate name in line %d: %s', i, name)

    cfg, component_type = util.re_key_value('type', cfg, optional=False)
    cfg, model_number = util.re_key_value('model', cfg, optional=False)
    try:
        model_number = int(model_number)
        specs[int(model_number)].append(name)
    except:
        raise ValueError('In line %d, expected integer for round: %s', i, line)

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
        lex_features = LexFeatures(e_corpus, f_corpus)

        cfg, min_count = util.re_key_value('min-count', cfg, optional=True)
        if min_count is None:
            min_count = 1
        else:
            min_count = int(min_count)

        cfg, max_count = util.re_key_value('max-count', cfg, optional=True)
        if max_count is None:
            max_count = -1
        else:
            max_count = int(max_count)

        feature_matrix = FeatureMatrix(e_corpus, f_corpus, lex_features,
                                       min_occurences=min_count, max_occurrences=max_count)
        dimensionality = feature_matrix.get_feature_size()
        logging.info('Unique lexical features: %d', dimensionality)

        cfg, init_option = util.re_key_value('init', cfg, optional=True)
        if init_option is None:
            init_option = 'normal'

        if init_option == 'uniform':
            weight_vector = np.full(dimensionality, 1.0 / dimensionality, dtype=np.float)
        else:  # random initialisation from a normal distribution with mean 0 and var 1.0
            weight_vector = np.random.normal(0, 1.0, dimensionality)

        cfg, sgd_steps = util.re_key_value('sgd-steps', cfg, optional=True)
        if sgd_steps is None:
            sgd_steps = 5
        else:
            sgd_steps = int(sgd_steps)

        cfg, sgd_attempts = util.re_key_value('sgd-attempts', cfg, optional=True)
        if sgd_attempts is None:
            sgd_attempts = 5
        else:
            sgd_attempts = int(sgd_attempts)

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

        else:
            if block_type == 'components':
                read_component(e_corpus, f_corpus, args, line, i, components, specs)
            elif block_type == 'iterations':
                read_iteration(line, i, n_iterations)

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
