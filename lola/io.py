"""
:Authors: - Wilker Aziz
"""
from lola.corpus import Corpus
from lola.corpus import CorpusView


def read_corpora(training_path: str,
                test_path: str,
                generating: bool,
                min_count: int, max_count: int) -> (CorpusView, CorpusView):
    """
    Return training and test data.

    :param training_path: path to training corpus
    :param test_path: path to test corpus (or None)
    :param generating: whether this is the side we are generating (French)
    :param min_count: minimum frequency for word to be retained in the vocabulary
    :param max_count: maximum frequency for word to be retained in the vocabulary
    :return: Training view and test view
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


def print_moses_format(alignments, ostream, skip_null=True):
    """
    Print alignments for a sentence in Moses format.
    :param s: sentence id (ignored)
    :param alignments: vector of m alignments
    :param posterior: posterior probabilities (ignored)
    :param ostream: where we print alignments to
    :param skip_null: ignores NULL alignments (default)
    """
    if skip_null:
        print(' '.join(['{0}-{1}'.format(i, j + 1) for j, i in enumerate(alignments) if i != 0]), file=ostream)
    else:
        print(' '.join(['{0}-{1}'.format(i, j + 1) for j, i in enumerate(alignments)]), file=ostream)


def print_lola_format(sid, alignments, posterior, e_corpus: Corpus, f_corpus: Corpus, ostream):
    """
    Print alignment in a human readable format.

    :param e_corpus: data we condition on
    :param f_corpus: data we generate
    :param sid: sentence id
    :param alignments: alignments (sequence of a_j values for each j)
    :param posterior: posterior p(a_j|f,e)
    :param ostream: where to write alignments to
    :return:
    """
    e_snt = e_corpus.sentence(sid)
    f_snt = f_corpus.sentence(sid)
    print(' '.join(['{0}:{1}|{2}:{3}|{4:.2f}'.format(j + 1,
                                                 f_corpus.translate(f_snt[j]),
                                                 i,
                                                 e_corpus.translate(e_snt[i]),
                                                 p)
                    for j, (i, p) in enumerate(zip(alignments, posterior))]),
          file=ostream)


def print_naacl_format(s, alignments, posterior, ostream, print_posterior=False, ids=None, skip_null=True):
    """
    Print alignments for a sentence in NAACL format.
    :param s: 0-based sentence id
    :param alignments: vector of m alignments
    :param posterior: posterior probability of each alignment
    :param ostream: where we print alignments to
    :param print_posterior: whether or not we display posterior probabilities (defaults to False)
    :param ids: if provided, overwrite sentence ids
    :param skip_null: ignores NULL alignment (default)
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
