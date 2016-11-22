"""
:Authors: - Wilker Aziz
"""
import theano
import theano.tensor as T
import numpy as np
from numpy import array as nparray
from theano.tensor.var import TensorVariable
from theano.tensor.elemwise import Elemwise
from typing import List, Iterable


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

#class MLPComponent(GenerativeComponent):
#    def __init__(self, name: str):
#        super(MLPComponent, self).__init__(name)
#        self._cpds = mlp_output(X)

#    def get(self, e_snt, f_snt, i, j):
#        return self._cpds[e_snt[i], f_snt[j]]



def train_latent_mlp(e_corpus: Corpus, f_corpus: Corpus, mlp: MLP):

    S = e_corpus.n_sentences()
    n_pairs = e_corpus.vocab_size() * f_corpus.vocab_size()
    n_input = e_corpus.vocab_size()
    n_output = f_corpus.vocab_size()
    print('pairs=%d input=%d output=%d' % (n_pairs, n_input, n_output))
    print('MLP(input=%d, output=%d)' % (mlp.n_input, mlp.n_output))

    X = np.identity(n_input, dtype=theano.config.floatX)

    print('Conditioning contexts')
    print(X)
    print()

    #shared_X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
    #shared_Y = theano.shared(np.asarray(Y, dtype=theano.config.floatX), borrow=True)

    # Create Theano variables for the MLP input
    mlp_input = T.matrix('mlp_input')
    # ... and the desired output
    #mlp_target = T.vector('mlp_target')
    mlp_expected = T.matrix('mlp_expected')
    mlp_cost = T.matrix('mlp_cost')
    # Learning rate and momentum hyperparameter values
    # Again, for non-toy problems these values can make a big difference
    # as to whether the network (quickly) converges on a good local minimum.
    learning_rate = 0.01
    momentum = 0

    # Create a theano function for computing the MLP's output given some input
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))


    # Create a function for computing the cost of the network given an input
    #cost = mlp.squared_error(mlp_input, mlp_target)
    cost = - mlp.expected_logprob(mlp_input, mlp_expected)
    mlp_cost = theano.function([mlp_input, mlp_expected], mlp.expected_logprob2(mlp_input, mlp_expected))
    #cost = mlp.logprob(mlp_input)
    # Create a theano function for training the network
    # train(mlp_input, mlp_target)
    train = theano.function([mlp_input, mlp_expected], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))

    # Keep track of the number of training iterations performed
    iteration = 0
    # We'll only train the network with 20 iterations.
    # A more common technique is to use a hold-out validation set.
    # When the validation error starts to increase, the network is overfitting,
    # so we stop training the net.  This is called "early stopping", which we won't do here.
    max_iteration = 100

    # build the categoricals by quering for all contexts
    theta = mlp_output(X)
    print('T-TABLE')
    print_ttable(e_corpus, f_corpus, theta)
    print()
    print('marginal_likelihood=%f\n' % marginal_likelihood(e_corpus, f_corpus, theta))

    while iteration < max_iteration:
        # Train the network using the entire training set.
        # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
        # where only a subset (or a single point) of the training set is used at each iteration.

        # compute expected counts (and give it to cost function)
        expected_counts = np.zeros((n_input, n_output), dtype=theano.config.floatX)
        for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
            joint = np.zeros((e_snt.shape[0], f_snt.shape[0]))
            # compute posterior for a_j = i
            for i, e in enumerate(e_snt):
                for j, f in enumerate(f_snt):
                    joint[i, j] = 1.0 / e_snt.shape[0] * theta[e, f]  # alignment * lexical (more components?)
            marginal = joint.sum(0)
            posterior = joint / marginal
            # accumulate expected counts for e, f pairs in the sentence
            for i, e in enumerate(e_snt):
                for j, f in enumerate(f_snt):
                    expected_counts[e, f] += posterior[i, j]

        print('EXPECTED-COUNTS %d' % iteration)
        print_ttable(e_corpus, f_corpus, expected_counts)
        print()
        print(mlp_cost(X, expected_counts))

        # This can also help the network to avoid local minima.
        current_cost = train(X, expected_counts)

        # Get the current network output for all points in the training set
        # update the categoricals
        theta = mlp_output(X)
        print('T-TABLE')
        print_ttable(e_corpus, f_corpus, theta)
        print('marginal_likelihood=%f\n' % marginal_likelihood(e_corpus, f_corpus, theta))
        iteration += 1

        print('ITERATION%d' % iteration)
        for e_snt, f_snt in zip(e_corpus.itersentences(), f_corpus.itersentences()):
            A = np.zeros(f_snt.shape[0], dtype=int)
            for j, f in enumerate(f_snt):
                joint = np.zeros(e_snt.shape[0])
                for i, e in enumerate(e_snt):
                    joint[i] = 1.0 / e_snt.shape[0] * theta[e, f]
                A[j] = joint.argmax()
            print(' '.join('%s(%d=%s)' % (f_corpus.translate(f_snt[j]), aj, e_corpus.translate(e_snt[aj])) for j, aj in enumerate(A)))

        #current_output = mlp_output(X)
        # We can compute the accuracy by thresholding the output
        # and computing the proportion of points whose class match the ground truth class.
        #accuracy = np.mean((current_output > .5) == y)
        # Plot network output after this iteration


def main(e_path, f_path):
    e_corpus = Corpus(e_path, null='<null>')
    f_corpus = Corpus(f_path)
    rng = np.random.RandomState(1234)
    builder = MLPBuilder(rng)
    builder.add_layer(e_corpus.vocab_size(), 3)
    builder.add_layer(3, f_corpus.vocab_size(), activation=T.nnet.softmax)
    mlp = builder.build()
    train_latent_mlp(e_corpus, f_corpus, mlp)



if __name__ == '__main__':
    main('../data/1k.e', '../data/1k.f')
    # test_mlp()