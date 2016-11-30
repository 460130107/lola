"""
:Authors: - Wilker Aziz
"""
import sys
from lola.corpus import Corpus
import lola.joint.cat as cat
from lola.joint.mlp import MLPLexical
import logging
from functools import partial
from lola.joint.conditional import EM, ConditionalModel, map_decoder
from lola.io import print_lola_format


def get_mlp_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    PL = cat.LengthDistribution()
    PM = cat.LengthDistribution()
    PAj = cat.UniformAlignment()
    PFj = MLPLexical(e_corpus, f_corpus)
    return ConditionalModel(PL, PM, PAj, PFj)


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    model = get_mlp_ibm1(e_corpus, f_corpus)

    EM(e_corpus, f_corpus, model, iterations=10)

    map_decoder(e_corpus, f_corpus, model,
                partial(print_lola_format,
                        e_corpus=e_corpus,
                        f_corpus=f_corpus,
                        ostream=sys.stdout))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')