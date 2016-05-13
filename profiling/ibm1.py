from lola.corpus import Corpus
from lola.hmm0 import EM, viterbi_alignments
from lola.basic_ibm import IBM1
from lola.component import LexicalParameters
import logging


def main(f_path, e_path, iterations=5, viterbi=True):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    f_corpus = Corpus(f_path)
    e_corpus = Corpus(e_path, null='<NULL>')
    ibm1 = IBM1(LexicalParameters(e_corpus.vocab_size(),
                                  f_corpus.vocab_size(),
                                  p=1.0 / f_corpus.vocab_size()))
    ibm1 = EM(e_corpus, f_corpus, iterations, ibm1)

    if viterbi:  # Viterbi alignments
        viterbi_alignments(e_corpus, f_corpus, ibm1)


if __name__ == '__main__':
    main('../training/5k.f', '../training/5k.e', 3, False)
