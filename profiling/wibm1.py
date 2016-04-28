from lola.corpus import Corpus
from lola.wibm1 import ibm1

def main(f_path, e_path, iterations=10, viterbi=True):
    f_corpus = Corpus(f_path)
    e_corpus = Corpus(e_path, null='<NULL>')
    ibm1(f_corpus, e_corpus, iterations, viterbi)


if __name__ == '__main__':
    main('../training/5k.f', '../training/5k.e', 3, False)
