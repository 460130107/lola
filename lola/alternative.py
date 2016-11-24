"""
:Authors: - Wilker Aziz
"""
import numpy as np
from collections import defaultdict
import logging
import itertools

"""
class RandomVariable:

    def __init__(self, name, *args):
        self.signature = (name,) + args

    def name(self):
        return self.signature[0]

    def __str__(self):
        return self.signature[0] if len(self.signature) == 1 \
            else '%s_%s' % (self.signature[0], ','.join(self.signature[1:]))
"""


class RandomVariable:

    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    def signature(self):
        return self.name, self.index

    def __str__(self):
        return '%s_%d' % (self.name, self.index)


class Component:

    def __init__(self, rv_name: str):
        self.rv_name = rv_name

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        """
        Generates a certain assignment of the random variable in context.
        Returns the contribution to the joint distribution.

        :param rv:
        :param value:
        :param context:
        :return:
        """
        pass

    def observe(self, rv: RandomVariable, value, context: dict, posterior: float):
        """
        Observe an assignment of the random variable with a certain posterior probability.

        :param rv:
        :param value:
        :param context:
        :param posterior:
        :return:
        """
        pass

    def update(self):
        pass


class UniformAlignment(Component):

    def __init__(self, rv_name='A', length_name='L'):
        super(UniformAlignment, self).__init__(rv_name)
        self._length_name = length_name

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        l = context[(self._length_name, None)]
        return 1.0 / l

    def observe(self, rv: RandomVariable, e, context: dict, posterior: float):
        pass

    def update(self):
        pass


class UnigramMixture(Component):

    def __init__(self, n_components, vocab_size, rv_name='E', comp_name='Z'):
        super(UnigramMixture, self).__init__(rv_name)
        self._cpds = np.ones((n_components, vocab_size), dtype=float)
        self._cpds /= vocab_size
        self._counts = np.zeros((n_components, vocab_size), dtype=float)
        self._comp_name = comp_name

    def generate(self, rv: RandomVariable, e, context: dict) -> float:
        z = context[(self._comp_name, None)]
        return self._cpds[z, e]

    def observe(self, rv: RandomVariable, e, context: dict, posterior: float):
        z = context[(self._comp_name, None)]
        self._counts[z, e] += posterior

    def update(self):
        normalisers = self._counts.sum(1)
        self._counts /= normalisers[:, np.newaxis]
        self._cpds = self._counts
        self._counts = np.zeros(self._counts.shape, dtype=float)


class TranslationDistribution(Component):

    def __init__(self, e_vocab_size=100, f_vocab_size=100, rv_name='F', e_name='E', a_name='A'):
        super(TranslationDistribution, self).__init__(rv_name)
        self._cpds = np.ones((e_vocab_size, f_vocab_size), dtype=float)
        self._cpds /= f_vocab_size
        self._counts = np.zeros((e_vocab_size, f_vocab_size), dtype=float)
        self._e_name = e_name
        self._a_name = a_name

    def generate(self, rv: RandomVariable, f, context: dict) -> float:
        j = rv.index
        aj = context[(self._a_name, j)]
        e = context[(self._e_name, aj)]
        return self._cpds[e, f]

    def observe(self, rv: RandomVariable, f, context: dict, posterior: float):
        j = rv.index
        aj = context[(self._a_name, j)]
        e = context[(self._e_name, aj)]
        self._counts[e, f] += posterior

    def update(self):
        normalisers = self._counts.sum(1)
        self._counts /= normalisers[:, np.newaxis]
        self._cpds = self._counts
        self._counts = np.zeros(self._counts.shape, dtype=float)


class Model:

    def __init__(self, components):
        self._components = tuple(components)
        self._comps_by_rv = defaultdict(list)
        for comp in self._components:
            self._comps_by_rv[comp.rv_name].append(comp)

    def generate(self, rv: RandomVariable, value, context: dict, state: dict) -> float:
        """
        Generates a variable returning its factor.
        The context is updated at the end of the method (after generation).
        :param rv:
        :param value:
        :param context:
        :return:
        """
        factor = 1.0
        for comp in self._comps_by_rv.get(rv.name, []):
            factor *= comp.generate(rv, value, context)
        state[rv.signature()] = value
        return factor

    def observe(self, rv: RandomVariable, value, context: dict, state: dict, posterior: float):
        """
        Observe a variable updating sufficient statistics.
        The context is updated at the end of the method (after observation).

        :param rv:
        :param value:
        :param context:
        :param posterior:
        :return:
        """
        for comp in self._comps_by_rv.get(rv.name, []):
            comp.observe(rv, value, context, posterior)
        state[rv.signature()] = value

    def update(self):
        for comp in self._components:
            comp.update()


from lola.corpus import Corpus


def marginal_likelihood(e_corpus: Corpus, f_corpus: Corpus, model: Model):
    total = 0.0
    for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
        context = {}
        # observations
        for i in range(e_snt.shape[0]):
            context[RandomVariable('E', i).signature()] = e_snt[i]
        context[RandomVariable('L', None).signature()] = e_snt.shape[0]
        context[RandomVariable('M', None).signature()] = f_snt.shape[0]
        # 0-order alignments
        for j in range(f_snt.shape[0]):
            marginal = 0.0
            for i in range(e_snt.shape[0]):
                p = model.generate(RandomVariable('A', j), i, context, context)
                p *= model.generate(RandomVariable('F', j), f_snt[j], context, context)
                marginal += p
            total += np.log(marginal)
    return - total / e_corpus.n_sentences()


def ibm1(e_corpus: Corpus, f_corpus: Corpus):

    components = [UniformAlignment(),
                  TranslationDistribution(e_corpus.vocab_size(),
                                          f_corpus.vocab_size())]
    model = Model(components)
    likelihood_aj = np.zeros(e_corpus.max_len())

    logging.info('Iteration %d Likelihood %f', 0, marginal_likelihood(e_corpus, f_corpus, model))

    for iteration in range(1, 6):

        # E-step
        for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
            context = {}
            # observations
            for i in range(e_snt.shape[0]):
                context[RandomVariable('E', i).signature()] = e_snt[i]
            context[RandomVariable('L', None).signature()] = e_snt.shape[0]
            context[RandomVariable('M', None).signature()] = f_snt.shape[0]

            # 0-order alignments
            for j in range(f_snt.shape[0]):
                marginal = 0.0
                for i in range(e_snt.shape[0]):
                    p = model.generate(RandomVariable('A', j), i, context, context)
                    p *= model.generate(RandomVariable('F', j), f_snt[j], context, context)
                    likelihood_aj[i] = p
                    marginal += p

                for i in range(e_snt.shape[0]):
                    posterior = likelihood_aj[i] / marginal
                    model.observe(RandomVariable('A', j), i, context, context, posterior)
                    model.observe(RandomVariable('F', j), f_snt[j], context, context, posterior)

        # M-step
        model.update()

        logging.info('Iteration %d Likelihood %f', iteration, marginal_likelihood(e_corpus, f_corpus, model))


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))
    ibm1(e_corpus, f_corpus)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')