import numpy as np
import collections as c


def open_file(file_name):
    f = open(file_name, 'rb')
    return f


def word2int(corpus_file):
    '''
    Translates a sentence to a numpy array and a word to an int.
    :param corpus_file:
    :return:
    '''
    #worddict = c.defaultdict(int)
    worddict = {}
    current_int = 0
    corpus = []
    for line in corpus_file:
        line = line.split()
        sentence = np.zeros(len(line), dtype=int)
        word_pos = 0
        for word in line:
            # if not worddict.__getitem__(word):
            if word not in worddict:
                worddict[word] = current_int
                current_int += 1
            word_int = worddict[word]
            sentence[word_pos] = word_int
            word_pos += 1
        corpus.append(sentence)
    return corpus, worddict

# f = open_file('training\example.f')
# corpus, worddict = word2int(f)
# print(corpus)
# print(worddict)