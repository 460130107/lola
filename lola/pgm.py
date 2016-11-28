"""
A random variable here is a tuple, with a name (str, first element) and perhaps additional fields.
The semantics of additional fields is up to the developer. I am not fixing it.

Examples:
    rv called Z:
        ('Z',)

    rv called Fj where j = 1:
        ('Fj', 1)

An assignment to a rv is a tuple made of the rv (first element) and its value.
The value can be whatever the developer likes, I again not fixing its semantics.

Examples:
    rv Z whose value is 1
        (('Z',), 1)
    rv Fj where j=1 whose value is "dog"
        (('Fj', 1), "dog")
    You can use helper functions:
        assign_rv(make_rv('Z'), 1))
        assign_rv(make_rv('Fj', j), "dog"))

"""
from collections import defaultdict
from typing import Tuple


RandomVariable = Tuple[str, int]


def make_rv(name, *args) -> RandomVariable:
    return (name,) + args


def assign_rv(rv: RandomVariable, value):
    return rv, value


class GenerativeComponent:
    """
    A locally normalised component.

    Example:

        a Component might be a lexical translation distribution that generates 'Fj' variables.
        then it's rv_name is 'Fj'
    """

    def __init__(self, rv_name: str):
        """
        The random variable it generates. For now this has to be a single random variable.
        Unless you create an rv whose semantics are different.
        TODO: deal with rv_names: list ?

        :param rv_name: str, the name of the random variable.
        """
        self.rv_name = rv_name

    def generate(self, rv: RandomVariable, value, context: dict) -> float:
        """
        Generates a certain assignment of the random variable in context.
        Returns the contribution to the joint distribution.

        :param rv: this is the rv, remember always a tuple, first the name, then the rest
        :param value: value of the rv
        :param context: context containing assignments to other rvs which this component may condition on
        :return: the probability (always locally normalised)
        """
        pass

    def observe(self, rv: RandomVariable, value, context: dict, posterior: float):
        """
        Observe an assignment of the random variable with a certain posterior probability.
        This is typically used in some sort of E-step.

        :param rv: this is the rv, remember always a tuple, first the name, then the rest
        :param value: value of the rv
        :param context: context containing assignments to other rvs which this component may condition on
        :param posterior: posterior probability of relevant latent variables
        """
        pass

    def update(self):
        """
        Update simply maximises the component based on observed expected counts.
        This is typically used in some sort of M-step.
        """
        pass


class DirectedModel:
    """
    A container for generative components.
    """

    def __init__(self, components):
        self._components = tuple(components)
        self._comps_by_rv = defaultdict(list)
        for comp in self._components:
            self._comps_by_rv[comp.rv_name].append(comp)
        for rv_name, comps in self._comps_by_rv.items():
            if len(comps) > 1:
                raise ValueError('In a directed model, variables should never be generated twice: %s' % rv_name)

    def __iter__(self):
        return iter(self._components)

    def generate_rv(self, rv: RandomVariable, value, context: dict, state: dict) -> float:
        """
        Generates a variable returning its factor.
        The context is updated at the end of the method (after generation).
        :param rv: a single rv
        :param value: its assignment
        :param context: the context in which it is generated
        :param state: where we store the rv assignment after generation
        :return: probability
        """
        factor = 1.0
        for comp in self._comps_by_rv.get(rv[0], []):  # we filter by rv's name
            factor *= comp.generate(rv, value, context)
        state[rv] = value
        return factor

    def observe_rv(self, rv: RandomVariable, value, context: dict, state: dict, posterior: float):
        """
        Observe a variable updating sufficient statistics.
        The context is updated at the end of the method (after observation).

        :param rv: a single rv
        :param value: its assignment
        :param context: the context in which it is generated
        :param state: where we store the rv assignment after generation
        :param posterior: the posterior over latent variables
        """
        for comp in self._comps_by_rv.get(rv[0], []):
            comp.observe(rv, value, context, posterior)
        state[rv] = value

    def generate_rvs(self, predictions: list, context: dict) -> (float, dict):
        """

        :param predictions: an ordered list of rv assingments, i.e., rvs paired with their values
            rvs get generated in the given order.
        :param context: context in which they get generated
        :return: probability, state (which in this case includes the context)
        """
        state = dict(context)
        factor = 1.0
        for rv, value in predictions:
            # update components associated with an RV
            for comp in self._comps_by_rv.get(rv[0], []):
                factor *= comp.generate(rv, value, state)
            state[rv] = value
        return factor, state

    def observe_rvs(self, predictions: list, context: dict, posterior: float) -> dict:
        """

        :param predictions: an ordered list of rv assingments, i.e., rvs paired with their values
            rvs get generated in the given order.
        :param context: context in which they get generated
        :param posterior: posterior over latent variables
        :return: state (which in this case includes the context)
        """
        state = dict(context)
        for rv, value in predictions:
            # update components associated with an RV
            for comp in self._comps_by_rv.get(rv[0], []):
                comp.observe(rv, value, state, posterior)
            state[rv] = value
        return state

    def update(self):
        for comp in self._components:
            comp.update()



