import numpy as np
import lola.read_data as rd


def ibm1(french, english, init_val, iterations):
    '''

    :param french:
    :param english:
    :param init_val:
    :param iterations:
    :return:
    '''
    french = rd.open_file(french)
    english = rd.open_file(english)
    corpus_french, dict_french = rd.word2int(french)
    corpus_english, dict_english = rd.word2int(english)
    lex_param = set_lex_param(init_val, dict_french, dict_english)
    for i in range(0, iterations):
        lex_counts, lex_normalize = e_step(corpus_french, corpus_english, lex_param, dict_french, dict_english)
        lex_param = m_step(lex_counts, lex_normalize)
    return lex_param


def e_step(french, english, lex_param, dict_french, dict_english):
    '''
    :param french:
    :param english:
    :param lex_param:
    :param dict_french:
    :param dict_english:
    :return:
    '''
    length_corpus = len(french)
    lex_counts = set_lex_counts(dict_french, dict_english)
    lex_normalize = set_normalize(dict_english)
    for sentence_index in range(0, length_corpus):
        english_sentence = english[sentence_index]
        french_sentence = french[sentence_index]
        for french_word in french_sentence:
            for english_word in english_sentence:
                tfe_total = sum_word(french_word, english_sentence, lex_param)
                delta = lex_param[english_word, french_word]/tfe_total
                lex_counts[english_word, french_word] += delta
                lex_normalize[english_word, 0] += delta
    return lex_counts, lex_normalize


def m_step(lex_counts, lex_normalize):
    '''

    :param lex_counts:
    :param lex_normalize:
    :return:
    '''
    lex_param = lex_counts / lex_normalize
    return lex_param


def set_lex_param(init_val, dict_french, dict_english):
    '''
    Initialize and return a t(f|e) parameter with uniform initializations.
    :param init_val:
    :param dict_french:
    :param dict_english:
    :return:
    '''
    length_french = len(dict_french)
    length_english = len(dict_english)
    lex_param = np.empty([length_english, length_french])
    lex_param.fill(init_val)
    return lex_param


def set_lex_counts(dict_french, dict_english):
    '''
    Initialize and return a count matrix with 0's
    :param dict_french:
    :param dict_english:
    :return:
    '''
    length_french = len(dict_french)
    length_english = len(dict_english)
    count_matrix = np.zeros((length_french, length_english))
    return count_matrix


def set_normalize(word_dict):
    '''

    :param word_dict:
    :return:
    '''
    length = len(word_dict)
    count_matrix = np.zeros((length, 1))
    return count_matrix


def sum_word(french_word, english_sentence, lex_param):
    '''

    :param french_word:
    :param english_sentence:
    :param lex_param:
    :return:
    '''
    total = 0
    for english_word in english_sentence:
        total += lex_param[english_word, french_word]
    return total


# french = 'training\hansards.36.2.f'
# english = 'training\hansards.36.2.e'
f = '../training/example.f'
e = '../training/example.e'

print(ibm1(f, e, 0.5, 5))
