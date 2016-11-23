"""
:Authors: - Wilker Aziz
"""
import logging
import theano
import theano.tensor as T
import numpy as np
from numpy import array as nparray
from theano.tensor.var import TensorVariable
from theano.tensor.elemwise import Elemwise
from typing import List, Iterable
from lola.util import re_key_value


def make_random_matrix(rng, n_input, n_output, activation=T.tanh):
    """
    From http://deeplearning.net/tutorial/mlp.html

    W is uniformely sampled from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden)),
     the output of uniform if converted using asarray to dtype
     theano.config.floatX so that the code is runable on GPU
     Note : optimal initialization of weights is dependent on the
            activation function used (among other things).
            For example, results presented in [Xavier10] suggest that you
            should use 4 times larger initial weights for sigmoid
            compared to tanh
            We have no info for other function, so we use the same as
            tanh.
    """
    W = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_input + n_output)),
            high=np.sqrt(6. / (n_input + n_output)),
            size=(n_input, n_output)
        ),
        dtype=theano.config.floatX
    )
    if activation and activation == theano.tensor.nnet.sigmoid:
        W *= 4
    return W


class Layer:

    def __init__(self, n_input: int, n_output: int, W: nparray, b: nparray, activation: Elemwise = T.tanh):
        """
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.

        :parameters:
            - rng: numpy random state
            - n_in: input dimensionality
            - n_out: output dimensionality
            - W: np.array, shape=(n_in, n_out)
                Optional weight matrix, if not given is initialised randomly.
            - b: np.array, shape=(n_out,)
                Optional bias vector, if not given is initialised randomly.
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        """
        assert W.shape == (n_input, n_output), \
            'W does not match the expected dimensionality (%d, %d) != %s' % (n_input, n_output, W.shape)
        assert b.shape == (n_output,), 'b does not match the expected dimensionality (%d,) != %s' % (n_output, b.shape)

        self.n_input = n_input
        self.n_output = n_output
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output,
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
        self.W = theano.shared(value=W.astype(theano.config.floatX),
                               # The name parameter is solely for printing purporses
                               name='W',
                               # Setting borrow=True allows Theano to use user memory for this object.
                               # It can make code slightly faster by avoiding a deep copy on construction.
                               # For more details, see
                               # http://deeplearning.net/software/theano/tutorial/aliasing.html
                               borrow=True)

        # We can force our bias vector b to be a column vector using numpy's reshape method.
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting (described below)
        self.b = theano.shared(value=b.astype(theano.config.floatX),
                               name='b',
                               borrow=True)

        self.activation = activation

        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]

    def output(self, x: TensorVariable) -> TensorVariable:
        """
        Compute this layer's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            Mixed, biased, and activated x
        """
        # Compute linear mix
        lin_output = T.dot(x, self.W) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return lin_output if self.activation is None else self.activation(lin_output)


class MLPBuilder:

    def __init__(self, rng):
        self._rng = rng
        self._layers = []  # type: List[Layer]

    def add_layer(self, n_input: int, n_output: int,
                  W: nparray = None, b: nparray = None,
                  activation: Elemwise = T.tanh):
        """

        :parameters:
            - n_input: input dimensionality
            - n_output: output dimensionality
            - W: np.array, shape=(n_input, n_output)
                Optional weight matrix (default: random initialisation)
            - b: np.array, shape=(n_input, n_output)
                Optional bias vector (default: 0s)
            - activation:
                Elementwise activation function (default: tanh)
        :returns:
            self
        """
        if self._layers:
            if self._layers[-1].n_output != n_input:
                i = len(self._layers)
                raise ValueError('Cannot wire hidden layers %d (n_output=%d) and %d (n_input=%d)' %
                                 (i, self._layers[-1].n_output, i + 1, n_input))
        if W is None:
            W = make_random_matrix(self._rng, n_input, n_output, activation)
        if b is None:
            b = np.zeros((n_output,), dtype=theano.config.floatX)
        self._layers.append(Layer(n_input, n_output, W, b, activation))
        return self

    def iterlayers(self) -> Iterable[Layer]:
        return iter(self._layers)

    def build(self) -> 'MLP':
        """Builds an MLP and resents the builder"""
        if not self._layers:
            raise ValueError('I cannot build an MLP without layers')
        mlp = MLP(self)
        self._layers = []
        return mlp


class MLP:

    def __init__(self, builder: MLPBuilder):
        """
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - builder : MLPBuilder
                A builder object that contains the configured layers.
        """
        # Initialize lists of layers
        self.layers = []  # type: List[Layer]
        for layer in builder.iterlayers():
            self.layers.append(layer)

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        self.n_input = self.layers[0].n_input
        self.n_output = self.layers[-1].n_output

    def output(self, x: TensorVariable) -> TensorVariable:
        """
        Compute the MLP's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        """
        for layer in self.layers:  # recursively transforms x
            x = layer.output(x)
        return x

    def squared_error(self, x: TensorVariable,
                      y: TensorVariable) -> TensorVariable:
        """
        Compute the squared euclidean error of the network output against the "true" output y

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        """
        return T.sum((self.output(x) - y) ** 2)

    def expected_logprob2(self, x: TensorVariable,
                         mu: TensorVariable) -> TensorVariable:
        """
        Compute the squared euclidean error of the network output against the "true" output y

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - mu : theano.tensor.var.TensorVariable
                Theano symbolic variable for expected level of output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        """
        return T.sum(T.mul(mu, T.log(self.output(x))), 1)

    def expected_logprob(self, x: TensorVariable,
                         mu: TensorVariable) -> TensorVariable:
        """
        Compute the squared euclidean error of the network output against the "true" output y

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - mu : theano.tensor.var.TensorVariable
                Theano symbolic variable for expected level of output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        """
        return T.sum(T.mul(mu, T.log(self.output(x))))

    def logprob(self, x: TensorVariable) -> TensorVariable:
        """
        Compute the squared euclidean error of the network output against the "true" output y

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - mu : theano.tensor.var.TensorVariable
                Theano symbolic variable for expected level of output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        """
        return T.sum(T.log(self.output(x)))


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    """
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    """
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        previous_step = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        step = momentum * previous_step - learning_rate * T.grad(cost, param)
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))
    return updates


from lola.corpus import Corpus
from itertools import product


def print_ttable(e_corpus: Corpus, f_corpus: Corpus, table: nparray):
    for e, f in product(range(e_corpus.vocab_size()), range(f_corpus.vocab_size())):
        print('context=%s decision=%s prob=%s' % (e_corpus.translate(e), f_corpus.translate(f), table[e, f]))


def marginal_likelihood(e_corpus: Corpus, f_corpus: Corpus, table: nparray):
    total = 0.0
    for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
        joint = np.zeros((e_snt.shape[0], f_snt.shape[0]))
        # compute posterior for a_j = i
        for i, e in enumerate(e_snt):
            for j, f in enumerate(f_snt):
                joint[i, j] = 1.0 / e_snt.shape[0] * table[e, f]  # alignment * lexical (more components?)
        marginal = joint.sum(0)
        total += np.log(marginal).sum()
    return total


from lola.component import GenerativeComponent
from lola.event import LexEventSpace
from lola.sparse import CPDTable
from lola.component import cmp_prob


class MLPComponent(GenerativeComponent):
    """
    This MLP component functions as a Categorical distribution.
    The input is an English word and the output is a distribution over the French vocabulary.

    """

    def __init__(self, e_corpus: Corpus,
                 f_corpus: Corpus,
                 name: str = "lexmlp",
                 rng=np.random.RandomState(1234),
                 hidden=[100],
                 learning_rate=0.1,
                 max_iterations=100,
                 patience=10,
                 patience_increase=2,
                 improvement_threshold=0.995):
        """

        :param e_corpus: data we condition on
        :param f_corpus: data we generate
        :param name: name of the component
        :param rng: numpy random state
        :param hidden: dimensionality of hidden layers
        :param learning_rate: initial learning rate
        :param max_iterations: maximum number of updates
        :param patience: minimum number of updates
        :param patience_increase:
        :param improvement_threshold:
        """
        super(MLPComponent, self).__init__(name, LexEventSpace(e_corpus, f_corpus))

        # TODO: generalise to batches?
        self._corpus_size = e_corpus.n_sentences()
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._patience = patience
        self._patience_increase = patience_increase
        self._improvement_threshold = improvement_threshold

        # The event space determines the input and output dimensionality
        self.n_input, self.n_output = self.event_space.shape
        # Input for the classifiers (TODO: should depend on the event space more closely)
        self._X = np.identity(self.n_input, dtype=theano.config.floatX)

        # Create MLP
        builder = MLPBuilder(rng)
        # ... the embedding layer
        builder.add_layer(self.n_input, hidden[0])
        # ... additional hidden layers
        for di, do in zip(hidden, hidden[1:]):
            builder.add_layer(di, do)
        # ... and the output layer (a softmax layer)
        builder.add_layer(hidden[-1], self.n_output, activation=T.nnet.softmax)
        self._mlp = builder.build()

        # Create Theano variables for the MLP input
        mlp_input = T.matrix('mlp_input')
        # ... and the expected output
        mlp_expected = T.matrix('mlp_expected')
        learning_rate = T.scalar('learning_rate')

        # Learning rate and momentum hyperparameter values
        # Again, for non-toy problems these values can make a big difference
        # as to whether the network (quickly) converges on a good local minimum.
        #learning_rate = 0.01
        momentum = 0

        # Create a theano function for computing the MLP's output given some input
        self._mlp_output = theano.function([mlp_input], self._mlp.output(mlp_input))

        # Create a function for computing the cost of the network given an input
        cost = - self._mlp.expected_logprob(mlp_input, mlp_expected)
        # Create a theano function for training the network
        self._train = theano.function([mlp_input, mlp_expected, learning_rate],
                                      # cost function
                                      cost,
                                      updates=gradient_updates_momentum(cost,
                                                                        self._mlp.params,
                                                                        learning_rate,
                                                                        momentum))

        # table to store the CPDs (output of MLP)
        self._cpds = self._mlp_output(self._X)
        # table to gather expected counts
        self._counts = np.zeros(self.event_space.shape, dtype=theano.config.floatX)
        self._i = 0

    @staticmethod
    def construct(e_corpus: Corpus, f_corpus: Corpus, name: str, config: str) -> dict:
        config, hidden = re_key_value('hidden', config, optional=True, default=[100])
        config, learning_rate = re_key_value('learning-rate', config, optional=True, default=0.1)
        config, max_iterations = re_key_value('max-iterations', config, optional=True, default=100)
        config, patience = re_key_value('patience', config, optional=True, default=10)
        config, patience_increase = re_key_value('patience-increase', config, optional=True, default=2)
        config, improvement_threshold = re_key_value('improvement-threshold', config, optional=True, default=0.995)
        config, seed = re_key_value('seed', config, optional=True, default=1234)
        return MLPComponent(e_corpus, f_corpus,
                            name=name,
                            rng=np.random.RandomState(seed),
                            hidden=hidden,
                            learning_rate=learning_rate,
                            max_iterations=max_iterations,
                            patience=patience,
                            patience_increase=patience_increase,
                            improvement_threshold=improvement_threshold)

    @staticmethod
    def example():
        return "lexmlp: type=MLPComponent hidden=[100] " \
               "learning-rate=0.1 max-iterations=100 patience=10 patience-increase=2 " \
               "improvement-threshold=0.995 seed=1234"

    def prob(self, e_snt: nparray, f_snt: nparray, i: int, j: int):
        return self._cpds[self.event_space.get(e_snt, f_snt, i, j)]

    def observe(self, e_snt: nparray, f_snt: nparray, i: int, j: int, p: float):
        self._counts[self.event_space.get(e_snt, f_snt, i, j)] += p

    def update(self):
        """
        A number of steps in the direction of the steepest decent,
         sufficient statistics are set to zero.
        """

        # First we need to scale the sufficient statistics by batch size
        self._counts /= self._corpus_size

        # We'll only train the network with 20 iterations.
        # A more common technique is to use a hold-out validation set.
        # When the validation error starts to increase, the network is overfitting,
        # so we stop training the net.  This is called "early stopping", which we won't do here.
        done_looping = False
        best_cost = np.inf
        best_iter = 0
        learning_rate = self._learning_rate
        patience = self._patience

        # TODO: implement adagrad
        for iteration in range(self._max_iterations):

            # Train the network using the entire training set.
            current_cost = self._train(self._X, self._counts, learning_rate)
            logging.debug('[%d] MLP cost=%s', iteration, current_cost)

            # Give it a chance to update cost and patience
            if current_cost < best_cost:
                if current_cost < best_cost * self._improvement_threshold:
                    if iteration >= patience - self._patience_increase:
                        patience += self._patience_increase
                best_cost = current_cost
                best_iter = iteration

            # Check patience
            if iteration > self._patience:
                logging.debug('Ran out of patience in iteration %d', iteration)
                break

        # Finally, we update the CPDs and reset the sufficient statistics to zero
        self._cpds = self._mlp_output(self._X)
        self._counts = np.zeros(self.event_space.shape, dtype=theano.config.floatX)

    def save(self, path):
        with open(path, 'w') as fo:
            for e, row in enumerate(self._cpds):
                for f, p in sorted(enumerate(row), key=cmp_prob):
                    e_str, f_str = self.event_space.readable((e, f))
                    print('%s %s %r' % (e_str, f_str, p), file=fo)

