import argparse
import sys
import logging

from lola.corpus import Corpus
from lola.hmm0 import EM, viterbi_alignments


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
                       default=10,
                       metavar='INT',
                       help='Number of iterations of IBM model 1')


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
        model = EM(f_corpus, e_corpus, args.ibm1, model_type='IBM1')
        viterbi_alignments(f_corpus, e_corpus, model)


if __name__ == '__main__':
    main()
