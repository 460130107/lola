import argparse
import sys
import logging
import numpy as np
import os

from lola.corpus import Corpus, CorpusView
import lola.hmm0 as hmm0
from lola.model import save_model
from functools import partial
from lola.config import configure
from lola.model import DefaultModel


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
    parser.add_argument('--merge', action='store_true', default=False,
                        help='Merge training and test set for training')

    cmd_naacl(parser.add_argument_group('NAACL Format'))
    cmd_logging(parser.add_argument_group('Logging'))

    return parser


def cmd_naacl(group):
    """Command line options for NAACL format"""
    group.add_argument('--naacl', action='store_true', default=False,
                       help='Print alignments in NAACL format (as well as Moses format)')
    group.add_argument('--posterior', action='store_true', default=False,
                       help='Print posterior probabilities for alignments')
    group.add_argument('--training-ids', metavar='FILE',
                        type=str,
                        help='Sentence ids for the training set (for NAACL format).')
    group.add_argument('--test-ids', metavar='FILE',
                        type=str,
                        help='Sentence ids for the test set (for NAACL format).')


def cmd_logging(group):
    """Command line options for output level"""
    group.add_argument('--viterbi', default=False, action='store_true',
                       help='Save Viterbi alignments')
    group.add_argument('--skip-null', action='store_true', default=False,
                       help='Skip NULL alignments when printing decisions')
    group.add_argument('--save-entropy', default=False, action='store_true',
                       help='Save empirical cross entropy progress')
    group.add_argument('--save-parameters', default=False, action='store_true',
                        help='Save parameters')
    group.add_argument('-v', '--verbose', default=0,
                        action='count',
                        help='Verbosity level')


def get_corpora(training_path, test_path, generating, min_count, max_count):
    """
    Return training and test data.

    :param training_path: path to training corpus
    :param test_path: path to test corpus (or None)
    :param generating: whether this is the side we are generating (French)
    :return:
    """
    if test_path is None:  # not test corpus
        if generating:
            corpus = Corpus(training_path, min_count=min_count, max_count=max_count)
        else:  # we are conditioning on this corpus
            corpus = Corpus(training_path, null='<NULL>', min_count=min_count, max_count=max_count)
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
            corpus = Corpus(lines, min_count=min_count, max_count=max_count)
        else:  # we are conditioning on this corpus
            corpus = Corpus(lines, null='<NULL>', min_count=min_count, max_count=max_count)
        # return two different views: the training view and the test view
        return CorpusView(corpus, 0, n_training), CorpusView(corpus, n_training, n_test)


def print_lex_parameter(e, f, p, e_corpus, f_corpus, ostream):
    print('{0} {1} {2}'.format(e_corpus.translate(e), f_corpus.translate(f), p), file=ostream)


def print_moses_format(s, alignments, posterior, skip_null, ostream):
    """
    Print alignments for a sentence in Moses format.
    :param s: 0-based sentence id
    :param alignments: vector of m alignments
    :param posterior: posterior probability of each alignment
    :param skip_null: whether or not we should print NULL alignments
    :param ostream: where we print alignments to
    """
    if skip_null:
        print(' '.join(['{0}-{1}'.format(i, j + 1) for j, i in enumerate(alignments) if i != 0]), file=ostream)
    else:
        print(' '.join(['{0}-{1}'.format(i, j + 1) for j, i in enumerate(alignments)]), file=ostream)


def print_naacl_format(s, alignments, posterior, skip_null, ostream, print_posterior=False, ids=None):
    """
    Print alignments for a sentence in NAACL format.
    :param s: 0-based sentence id
    :param alignments: vector of m alignments
    :param posterior: posterior probability of each alignment
    :param skip_null: whether or not we should print NULL alignments
    :param ostream: where we print alignments to
    :param ids: if provided, overwrite sentence ids
    """
    if ids is None:
        def get_id(_s):
            return _s + 1  # default to 1-based
    else:
        def get_id(_s):
            return ids[_s]  # use the ids provided

    if print_posterior:
        def get_string(_s, _i, _j, _p):  # print posterior as well
            return '{0} {1} {2} S {3}'.format(get_id(_s), _i, _j + 1, _p)
    else:
        def get_string(_s, _i, _j, _p):  # ignore posterior
            return '{0} {1} {2}'.format(get_id(_s), _i, _j + 1)

    if skip_null:
        def print_string(_s, _i, _j, _p):  # check for NULL alignments and skip them
            if _i != 0:
                print(get_string(_s, _i, _j, _p), file=ostream)
    else:
        def print_string(_s, _i, _j, _p):  # print everything
            print(get_string(_s, _i, _j, _p), file=ostream)

    for j, i in enumerate(alignments):
        print_string(s, i, j, posterior[j])


def print_moses_and_naacl_formats(s, alignments, posterior, skip_null, moses_stream, naacl_stream,
                                  print_posterior=False, ids=None):
    """Prints alignments for sentence s in both Moses and NAACL format."""
    print_moses_format(s, alignments, posterior, skip_null, moses_stream)
    print_naacl_format(s, alignments, posterior, skip_null, naacl_stream, print_posterior, ids)


def save_viterbi(e_corpus, f_corpus, ids, model, path, args):
    """
    Saves the Viterbi decisions in various formats.
    """

    if args.naacl:
        with open('{0}.moses'.format(path), 'w') as fm:
            with open('{0}.naacl'.format(path), 'w') as fn:
                hmm0.viterbi_alignments(e_corpus, f_corpus, model,
                                        callback=partial(print_moses_and_naacl_formats,
                                                         skip_null=args.skip_null,
                                                         moses_stream=fm,
                                                         naacl_stream=fn,
                                                         print_posterior=args.posterior,
                                                         ids=ids))
    else:
        with open('{0}.moses'.format(path), 'w') as fm:
            hmm0.viterbi_alignments(e_corpus, f_corpus, model,
                                    callback=partial(print_moses_format,
                                                     skip_null=args.skip_null,
                                                     ostream=fm))


def save_entropy(entropies, path):
    with open(path, 'w') as fo:
        for h in entropies:
            print(h, file=fo)


def train_and_apply(e_training, f_training, apply_to, iterations,
                     model: DefaultModel, model_name, args, initialiser=None):
    logging.info('Starting %d iterations of %s', iterations, model_name)
    # train it with EM for a number of iterations
    model, training_entropy = hmm0.EM(e_training, f_training, iterations, model)

    if args.save_entropy:  # save the entropy of each EM iteration
        save_entropy(training_entropy, '{0}/{1}.EM'.format(args.output, model_name))

    if args.save_parameters:
        logging.info('Saving parameters of %s', model_name)
        save_model(model, e_training, f_training, '{0}/{1}'.format(args.output, model_name))

    entropies = []
    # apply model to each parallel corpus
    for name, e_corpus, f_corpus, ids in apply_to:
        corpus_entropy = hmm0.empirical_cross_entropy(e_corpus, f_corpus, model)

        entropies.append(corpus_entropy)
        logging.info('%s %s set perplexity: %f', model_name, name, corpus_entropy)

        if args.viterbi:
            logging.info('Saving %s Viterbi decisions for %s', model_name, name)
            # apply model to training data
            save_viterbi(e_corpus, f_corpus, ids,
                         model,
                         '{0}/{1}.{2}.viterbi'.format(args.output, model_name, name),
                         args)

    return model, entropies


def pipeline(e_training, f_training, apply_to, args):

    logging.info('Constructing extractors and components')
    config = configure(args.config, e_training, f_training, args)
    import shutil
    shutil.copy(args.config, '{0}/config.ini'.format(args.output))

    # we start with all components
    components = config.components()

    for i, model_spec in enumerate(config.itermodels()):
        logging.info(str(model_spec))

        # then we make a model based on a certain selection of components
        model = model_spec.make(components)

        # we optimise this model for a number of EM iterations
        optimised, entropies = train_and_apply(e_training,
                                                f_training,
                                                apply_to,
                                                model_spec.iterations,
                                                model,
                                                model_spec.name,
                                                args)

        # then we update our old components
        # keeping optimised ones
        for comp_name, optimised_comp in zip(model_spec.components, optimised):
            components[comp_name] = optimised_comp


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
    f_training, f_test = get_corpora(args.french, args.test_f, True, args.min_f_count, args.max_f_count)
    e_training, e_test = get_corpora(args.english, args.test_e, False, args.min_e_count, args.max_e_count)  # If we forget generating=False things go really bad!

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
