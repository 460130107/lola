import argparse
import sys
import logging

from lola.corpus import Corpus, CorpusView
import lola.hmm0 as hmm0
from lola.component import LexicalParameters, JumpParameters, BrownDistortionParameters
from lola.basic_ibm import IBM1, VogelIBM2, BrownIBM2


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='lola')

    parser.description = 'Log-linear alignment models'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('-f', '--french', metavar='FILE',
                        type=str,
                        help='French corpus (the data we generate)')
    parser.add_argument('-e', '--english', metavar='FILE',
                        type=str,
                        help='English corpus (the data we condition on)')
    parser.add_argument('--test-f', metavar='FILE',
                        type=str,
                        help='A French test corpus (the date we generate)')
    parser.add_argument('--test-e', metavar='FILE',
                        type=str,
                        help='An English test corpus (the date we condition on)')
    parser.add_argument('--merge', action='store_true', default=False,
                        help='Merge training and test set for training')
    parser.add_argument('--lexparams', metavar='FILE',
                        type=str,
                        help='Save lexical parameters')
    parser.add_argument('--distortion-type', choices=['Vogel', 'Brown'],
                        type=str, default='Vogel',
                        help='Type of distortion parameters')

    cmd_estimation(parser.add_argument_group('Parameter estimation'))

    cmd_logging(parser.add_argument_group('Logging'))

    return parser


def cmd_estimation(group):
    group.add_argument('--ibm1',
                       type=int,
                       default=5,
                       metavar='INT',
                       help='Number of iterations of IBM model 1')
    group.add_argument('--ibm2',
                       type=int,
                       default=0,
                       metavar='INT',
                       help='Number of iterations of IBM model 2')


def cmd_logging(group):
    group.add_argument('--save-entropy',
                       type=str, metavar='FILE',
                       help='Save empirical cross entropy progress')
    group.add_argument('--viterbi', metavar='PATH',
                       type=str,
                       help='Save Viterbi alignments to a file')
    group.add_argument('-v', '--verbose', default=0,
                        action='count',
                        help='Verbosity level')


def get_corpora(training_path, test_path, generating):
    """
    Return training and test data.

    :param training_path: path to training corpus
    :param test_path: path to test corpus (or None)
    :param generating: whether this is the side we are generating (French)
    :return:
    """
    if test_path is None:  # not test corpus
        if generating:
            corpus = Corpus(training_path)
        else:  # we are conditioning on this corpus
            corpus = Corpus(training_path, null='<NULL>')
        return corpus, None
    else:
        # read training data
        with open(training_path, 'r') as fi:
            lines = fi.readlines()
        n_training = len(lines)
        # read test data
        with open(test_path, 'r') as fi:
            lines.extend(fi.readlines())
        n_test = len(lines) - n_training
        # create a big corpus with everything
        if generating:
            corpus = Corpus(lines)
        else:  # we are conditioning on this corpus
            corpus = Corpus(lines, null='<NULL>')
        # return two different views: the training view and the test view
        return CorpusView(corpus, 0, n_training), CorpusView(corpus, n_training, n_test)


def save_viterbi(e_corpus, f_corpus, model, path):
    with open(path, 'w') as fo:
        hmm0.viterbi_alignments(e_corpus, f_corpus, model, ostream=fo)


def save_entropy(entropies, path):
    with open(path, 'w') as fo:
        for h in entropies:
            print(h, file=fo)


def get_ibm1(e_corpus, f_corpus):
    return IBM1(LexicalParameters(e_corpus.vocab_size(),
                                  f_corpus.vocab_size(),
                                  p=1.0 / f_corpus.vocab_size()))


def get_ibm2(e_corpus, f_corpus, dist_type='Vogel'):
    if dist_type == 'Vogel':
        return VogelIBM2(LexicalParameters(e_corpus.vocab_size(),
                                           f_corpus.vocab_size(),
                                           p=1.0 / f_corpus.vocab_size()),
                         JumpParameters(e_corpus.max_len(),
                                        f_corpus.max_len(),
                                        1.0 / (e_corpus.max_len() + f_corpus.max_len() + 1)))
    elif dist_type == 'Brown':
        return BrownIBM2(LexicalParameters(e_corpus.vocab_size(),
                                           f_corpus.vocab_size(),
                                           p=1.0 / f_corpus.vocab_size()),
                         BrownDistortionParameters(e_corpus.max_len(),
                                                   1.0 / (e_corpus.max_len())))


def initial_model(e_corpus, f_corpus, model_type, dist_type, initialiser=None):
    if model_type == 'IBM1':
        model = get_ibm1(e_corpus, f_corpus)
    elif model_type == 'IBM2':
        model = get_ibm2(e_corpus, f_corpus, dist_type)
    else:
        raise ValueError('I do not know this type of model: %s' % model_type)

    if initialiser is not None:
        model.initialise(initialiser)

    return model


def train_and_apply(e_training, f_training, apply_to, iterations, model_type, args, initialiser=None):
    logging.info('Starting %d iterations of %s', iterations, model_type)
    # get an initial model (possibly based on previously trained models)
    model = initial_model(e_training, f_training, model_type, args.distortion_type, initialiser)
    # train it with EM for a number of iterations
    model, training_entropy = hmm0.EM(e_training, f_training, iterations, model)

    if args.save_entropy:  # save the entropy of each EM iteration
        save_entropy(training_entropy, '{0}.{1}.EM'.format(args.save_entropy, model_type))

    entropies = []
    # apply model to each parallel corpus
    for name, e_corpus, f_corpus in apply_to:
        corpus_entropy = hmm0.empirical_cross_entropy(e_corpus, f_corpus, model)

        entropies.append(corpus_entropy)
        logging.info('%s %s set perplexity: %f', model_type, name, corpus_entropy)

        if args.viterbi:
            logging.info('Saving %s Viterbi decisions for %s', model_type, name)
            # apply model to training data
            save_viterbi(e_corpus, f_corpus, model, '{0}.{1}.{2}.viterbi'.format(args.viterbi, model_type, name))

    return model, entropies


def pipeline(e_training, f_training, apply_to, args):

    if args.ibm1 > 0:
        ibm1, ibm1_entropies = train_and_apply(e_training,
                                               f_training,
                                               apply_to,
                                               args.ibm1,
                                               'IBM1',
                                               args)

        # TODO: save IBM1 entropies

    else:
        ibm1 = None

    if args.ibm2 > 0:
        initialiser = {}
        # configure initialisation
        if ibm1 is not None:
            initialiser['IBM1'] = ibm1

        ibm2, ibm2_entropies = train_and_apply(e_training,
                                               f_training,
                                               apply_to,
                                               args.ibm2,
                                               'IBM2',
                                               args,
                                               initialiser)
        # TODO: save IBM2 entropies



def main():
    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    # read corpora
    logging.info('Reading data')
    f_training, f_test = get_corpora(args.french, args.test_f, True)
    e_training, e_test = get_corpora(args.english, args.test_e, True)

    # we will always apply our model to the training corpus after EM
    apply_to = [('training', e_training, f_training)]

    if e_test and f_test:  # and if we have a test set, we also apply to it
        apply_to.append(('test', e_test, f_test))

    if args.merge:  # sometimes to avoid OOVs we merge training and test before EM
        # note that this is not cheating because the task remains unsupervised
        e_merged = e_training.underlying()
        f_merged = f_training.underlying()
    else:
        e_merged = e_training
        f_merged = f_training

    pipeline(e_merged, f_merged, apply_to, args)



if __name__ == '__main__':
    main()
