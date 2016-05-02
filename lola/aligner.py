import argparse
import sys
import logging

from lola.corpus import Corpus
import lola.hmm0 as hmm0
from lola.model import IBM1, IBM1ExpectedCounts
from lola.params import LexicalParameters
from lola.jump_ibm2 import IBM2, IBM2ExpectedCounts, JumpParameters


def EM(f_corpus, e_corpus, iterations, model_type, initialiser=None):
    if model_type == 'IBM1':
        model = IBM1(LexicalParameters(e_corpus.vocab_size(),
                                       f_corpus.vocab_size(),
                                       p=1.0/f_corpus.vocab_size()))
        suffstats = IBM1ExpectedCounts(e_corpus.vocab_size(),
                                       f_corpus.vocab_size())
    elif model_type == 'IBM2':
        model = IBM2(LexicalParameters(e_corpus.vocab_size(),
                                       f_corpus.vocab_size(),
                                       p=1.0/f_corpus.vocab_size()),
                     JumpParameters(e_corpus.max_len(),
                                    f_corpus.max_len(),
                                    1.0/(e_corpus.max_len() + f_corpus.max_len() + 1)))
        suffstats = IBM2ExpectedCounts(e_corpus.vocab_size(),
                                       f_corpus.vocab_size(),
                                       e_corpus.max_len(),
                                       f_corpus.max_len())
    else:
        raise ValueError('I do not know this type of model: %s' % model_type)

    if initialiser is not None:
        model.initialise(initialiser)

    return hmm0.EM(f_corpus, e_corpus, iterations, model, suffstats)


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='lola')

    parser.description = 'Log-linear alignment models'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('bitext', nargs='?', metavar='BITEXT',
                        type=argparse.FileType('r'), default=sys.stdin,
                        help='One sentence pair per line (French ||| English). '
                             'This is used only if -f and -e are not provided.')
    parser.add_argument('-f', '--french', metavar='FILE',
                        type=str,
                        help='French corpus (the data we generate)')
    parser.add_argument('-e', '--english', metavar='FILE',
                        type=str,
                        help='English corpus (the data we condition on)')
    parser.add_argument('--lexparams', metavar='FILE',
                        type=str,
                        help='Save lexical parameters')

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
    group.add_argument('--likelihood',
                       type=str, metavar='FILE',
                       help='Save log-likelihood progress')
    group.add_argument('-v', '--verbose', default=0,
                        action='count',
                        help='Verbosity level')


def main():
    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    # read corpus
    logging.info('Reading data')
    if args.french and args.english:
        f_corpus = Corpus(args.french)
        e_corpus = Corpus(args.english, null='<NULL>')
    else:
        f_stream = []
        e_stream = []
        for line in args.bitext:
            parts = line.strip().split(' ||| ')
            if len(parts) != 2:  # ignore unaligned sentence pairs
                continue
            f_stream.append(parts[0])
            e_stream.append(parts[1])
        f_corpus = Corpus(f_stream)
        e_corpus = Corpus(e_stream, null='<NULL>')

    if args.ibm1 > 0:
        logging.info('Starting %d iterations of IBM model 1', args.ibm1)
        ibm1 = EM(f_corpus, e_corpus, args.ibm1, model_type='IBM1')
        if args.ibm2 == 0:
            hmm0.viterbi_alignments(f_corpus, e_corpus, ibm1)
    else:
        ibm1 = None

    if args.ibm2 > 0:
        logging.info('Starting %d iterations of IBM model 2', args.ibm2)
        initialiser = {}
        if ibm1 is not None:
            initialiser['IBM1'] = ibm1
        ibm2 = EM(f_corpus, e_corpus, args.ibm2, model_type='IBM2', initialiser=initialiser)
        hmm0.viterbi_alignments(f_corpus, e_corpus, ibm2)

if __name__ == '__main__':
    main()
