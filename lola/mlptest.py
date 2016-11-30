"""
:Authors: - Wilker Aziz
"""
from lola.corpus import Corpus
from lola.models import zero_order_joint_model
import lola.cat as cat
from lola.mlp import MLPLexical
import logging


def get_mlp_ibm1(e_corpus: Corpus, f_corpus: Corpus):
    PL = cat.LengthDistribution()
    PM = cat.LengthDistribution()
    PZ = cat.ClusterDistribution(1)
    PEi = cat.TargetDistribution()
    PAj = cat.UniformAlignment()
    PFj = MLPLexical(e_corpus, f_corpus)
    return PL, PM, PZ, PEi, PAj, PFj


def main(e_path, f_path):

    e_corpus = Corpus(open(e_path), null='<null>')
    f_corpus = Corpus(open(f_path))

    PL, PM, PZ, PEi, PAj, PFj = get_mlp_ibm1(e_corpus, f_corpus)

    zero_order_joint_model(e_corpus, f_corpus,
                           PL, PM, PZ, PEi, PAj, PFj,
                           iterations=10)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    main('example.e', 'example.f')