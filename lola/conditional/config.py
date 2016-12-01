"""
:Authors: - Wilker Aziz
"""

import lola.util as util
from lola.corpus import Corpus
from lola.conditional.mlp import MLPComponent
from lola.conditional.lr import LRComponent
from lola.conditional.model import GenerativeModel
from lola.conditional.component import BrownLexical
from lola.conditional.component import UniformAlignment
from lola.conditional.component import VogelJump


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

    def make(self, components_repo) -> GenerativeModel:
        if not all(name in components_repo for name in self.components):
            raise ValueError('Missing components')
        return GenerativeModel([components_repo[name] for name in self.components])


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


def read_component(e_corpus: Corpus, f_corpus: Corpus, args, line: str, i: int, state: Config):
    """
    Instantiate components.
    If you contribute a new component, make sure to construct it here.

    :param e_corpus:
    :param f_corpus:
    :param args:
    :param line:
    :param i:
    :param state:
    :return:
    """
    try:
        cfg, [name, _] = util.re_sub('^([^:]+)(:)', '', line)
    except:
        raise ValueError('In line %d, expected component name: %s' % (i, line))

    if state.has_component(name):
        raise ValueError('Duplicate component name in line %d: %s', i, name)

    cfg, component_type = util.re_key_value('type', cfg, optional=False, dtype=str)

    if component_type == 'BrownLexical':
        state.add_component(name, BrownLexical(e_corpus, f_corpus, name=name))
    elif component_type == 'UniformAlignment':
        state.add_component(name, UniformAlignment(name=name))
    elif component_type == 'VogelJump':
        state.add_component(name, VogelJump(e_corpus.max_len(), name=name))
    elif component_type == "LexMLP":
        state.add_component(name, MLPComponent.construct(e_corpus, f_corpus, name, cfg))
    elif component_type == "LexLR":
        state.add_component(name, LRComponent.construct(e_corpus, f_corpus, name, cfg))
    else:
        raise ValueError("I do not know this type of generative component: %s" % component_type)


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
                        # '[extractors]': read_extractor,
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

    :param path: path to config file
    :param e_corpus: target Corpus (the conditioning event)
    :param f_corpus: source Corpus (the data we generate)
    :param args: command line arguments
    :return: Config
    """

    with open(path) as fi:
        return parse_blocks(e_corpus, f_corpus, args, fi, 1)


_EXAMPLE_ = """
# Here we declare generative components that can be used in building models
[components]
# <Name>: type=<Type>
lexical: type=BrownLexical
uniform: type=UniformAlignment
jump: type=VogelJump

[models]
# <Name>: iterations=<Number> components=['<Name1>','<Name2>']
ibm1: iterations=5 components=['lexical','uniform']
ibm2: iterations=5 components=['lexical','jump']
"""


def example():
    return _EXAMPLE_


