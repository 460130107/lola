"""
:Authors: - Wilker Aziz
"""



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