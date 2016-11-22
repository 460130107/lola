import argparse
import sys
import shutil
import logging
import numpy as np
import os

from functools import partial
import lola.em as em
from lola.config import configure
from lola.model import GenerativeModel
from lola.io import print_moses_format
from lola.io import print_naacl_format
from lola.io import print_lola_format
from lola.io import read_corpora


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='lola')

    parser.description = 'Log-linear alignment models'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('config', type=str,
                        help='Path to a model configuration specifying components, '
                             'feature extractors, EM iterations for each model, '
                             'and SGD options where applicable.')
    parser.add_argument('output', metavar='PATH',
                        type=str,
                        help='Output directory')
    parser.add_argument('-f', '--french', metavar='FILE',
                        type=str,
                        help='French corpus (the data we generate)')
    parser.add_argument('-e', '--english', metavar='FILE',
                        type=str,
                        help='English corpus (the data we condition on)')
    parser.add_argument('--test-f', metavar='FILE',
                        type=str,
                        help='A French test corpus (the data we generate)')
    parser.add_argument('--test-e', metavar='FILE',
                        type=str,
                        help='An English test corpus (the data we condition on)')
    parser.add_argument('--min-e-count', type=int, default=0)
    parser.add_argument('--max-e-count', type=int, default=0)
    parser.add_argument('--min-f-count', type=int, default=0)
    parser.add_argument('--max-f-count', type=int, default=0)

    cmd_viterbi(parser.add_argument_group('Viterbi alignments'))
    cmd_logging(parser.add_argument_group('Logging'))

    return parser


def cmd_viterbi(group):
    """Command line options for alignment formats"""
    group.add_argument('--moses', action='store_true', default=False,
                       help='Save alignments in Moses format')
    group.add_argument('--naacl', action='store_true', default=False,
                       help='Print alignments in NAACL format (as well as Moses format)')
    group.add_argument('--posterior', action='store_true', default=False,
                       help='Print posterior probabilities in NAACL file')
    group.add_argument('--training-ids', metavar='FILE',
                        type=str,
                        help='Sentence ids for the training set (for NAACL format).')
    group.add_argument('--test-ids', metavar='FILE',
                        type=str,
                        help='Sentence ids for the test set (for NAACL format).')


def cmd_logging(group):
    """Command line options for output level"""
    group.add_argument('--save-entropy', default=False, action='store_true',
                       help='Save empirical cross entropy progress')
    group.add_argument('--save-parameters', default=False, action='store_true',
                        help='Save parameters')
    group.add_argument('-v', '--verbose', default=0,
                        action='count',
                        help='Verbosity level')


def print_lex_parameter(e, f, p, e_corpus, f_corpus, ostream):
    print('{0} {1} {2}'.format(e_corpus.translate(e), f_corpus.translate(f), p), file=ostream)


def print_alignments(s, alignments, posterior, streams, e_corpus, f_corpus, print_posterior=False, ids=None):
    for fileformat, stream in streams.items():
        if fileformat == 'moses':
            print_moses_format(alignments, stream, skip_null=True)
        elif fileformat == 'naacl':
            print_naacl_format(s, alignments, posterior, stream,
                               print_posterior=print_posterior, ids=ids, skip_null=True)
        elif fileformat == 'lola':
            print_lola_format(s, alignments, posterior, e_corpus, f_corpus, stream)


def save_viterbi(e_corpus, f_corpus, ids, model, path, args):
    """
    Saves the Viterbi decisions in various formats.
    """
    streams = {'lola': open('{0}.lola'.format(path), 'w')}
    if args.naacl:
        streams['naacl'] = open('{0}.naacl'.format(path), 'w')
    if args.moses:
        streams['moses'] = open('{0}.moses'.format(path), 'w')

    em.viterbi_alignments(e_corpus, f_corpus, model,
                          callback=partial(print_alignments,
                                           streams=streams,
                                           e_corpus=e_corpus,
                                           f_corpus=f_corpus,
                                           print_posterior=args.posterior,
                                           ids=ids))


def save_entropy(entropies, path):
    with open(path, 'w') as fo:
        for h in entropies:
            print(h, file=fo)


def train_and_apply(e_training, f_training, apply_to, iterations,
                    model: GenerativeModel, model_name, args, initialiser=None):
    logging.info('Starting %d iterations of %s', iterations, model_name)
    # train it with EM for a number of iterations
    training_entropy = em.EM(e_training, f_training, iterations, model)

    if args.save_entropy:  # save the entropy of each EM iteration
        save_entropy(training_entropy, '{0}/{1}.EM'.format(args.output, model_name))

    if args.save_parameters:
        logging.info('Saving parameters of %s', model_name)
        model.save('{0}/{1}'.format(args.output, model_name))
        #save_model(model, e_training, f_training, '{0}/{1}'.format(args.output, model_name))

    entropies = []
    # apply model to each parallel corpus
    for name, e_corpus, f_corpus, ids in apply_to:
        corpus_entropy = em.empirical_cross_entropy(e_corpus, f_corpus, model)

        entropies.append(corpus_entropy)
        logging.info('%s %s set perplexity: %f', model_name, name, corpus_entropy)

        logging.info('Saving %s Viterbi decisions for %s', model_name, name)
        # apply model to training data
        save_viterbi(e_corpus, f_corpus, ids,
                     model,
                     '{0}/{1}.{2}.viterbi'.format(args.output, model_name, name),
                     args)

    return entropies


def pipeline(e_training, f_training, apply_to, args):
    """
    Trains a sequence of models. Components that are shared are reused from previous optimisations.

    :param e_training:
    :param f_training:
    :param apply_to:
    :param args:
    :return:
    """

    logging.info('Constructing extractors and components')
    config = configure(args.config, e_training, f_training, args)
    shutil.copy(args.config, '{0}/config.ini'.format(args.output))

    # we start with all components
    components_repo = config.components()

    # For each model specification
    for i, model_spec in enumerate(config.itermodels()):
        logging.info(str(model_spec))

        # then we make a model based on a certain selection of components
        model = model_spec.make(components_repo)

        # we optimise this model for a number of EM iterations
        train_and_apply(e_training, f_training, apply_to,
                        model_spec.iterations, model, model_spec.name, args)

        # then we update our old components
        # keeping optimised ones
        for comp in model:
            components_repo[comp.name] = comp


def main():
    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # read corpora
    logging.info('Reading data')
    f_training, f_test = read_corpora(args.french, args.test_f, True, args.min_f_count, args.max_f_count)
    # If we forget generating=False things go really bad!
    e_training, e_test = read_corpora(args.english, args.test_e, False, args.min_e_count, args.max_e_count)

    logging.info('Vocabulary size: English %d x French %d', e_training.vocab_size(), f_training.vocab_size())

    # read ids (sometimes necessary for NAACL format)
    if args.training_ids:
        training_ids = np.loadtxt(args.training_ids, dtype=int)
    else:
        training_ids = None
    if args.test_ids:
        test_ids = np.loadtxt(args.test_ids, dtype=int)
    else:
        test_ids = None

    # we will always apply our model to the training corpus after EM
    apply_to = [('training', e_training, f_training, training_ids)]

    if e_test and f_test:  # and if we have a test set, we also apply to it
        apply_to.append(('test', e_test, f_test, test_ids))

    # to avoid OOVs we merge training and test before EM
    # note that this is not cheating because the task remains unsupervised
    e_merged = e_training.underlying()
    f_merged = f_training.underlying()

    pipeline(e_merged, f_merged, apply_to, args)


if __name__ == '__main__':
    main()
