"""
:Authors: - Wilker Aziz
"""
from lola.corpus import Corpus
from lola.models import EM
from lola.models import Model
from lola.models import map_decoder
import lola.cat as cat
from lola.mlp import MLPLexical
import logging
from functools import partial
from lola.models import print_map


def get_mlp_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    PL = cat.LengthDistribution()
    PM = cat.LengthDistribution()
    PZ = cat.ClusterDistribution(1)
    PEi = cat.TargetDistribution()
    PAj = cat.UniformAlignment()
    PFj = MLPLexical(e_corpus, f_corpus)
    return Model(PL, PM, PZ, PEi, PAj, PFj)


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    model = get_mlp_ibm1(e_corpus, f_corpus)

    EM(e_corpus, f_corpus, model, iterations=10)

    map_decoder(e_corpus, f_corpus, model,
                partial(print_map,
                        e_corpus=e_corpus,
                        f_corpus=f_corpus))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')