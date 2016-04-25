import numpy as np

def open_file(fileName): 
	f = open(fileName, 'r')
	return f

# translates a sentence to a numpy array and a word to an int
def wordToInt(file):
	dict = {}
	currentInt = 0
	corpus = [] 
	for line in file:
		line = line.split()
		sentence = np.zeros(len(line))
		wordPos = 0
		for word in line:
			if  not dict.has_key(word):
				dict[word] = currentInt
				currentInt += 1
			wordInt = dict[word]
			sentence[wordPos] = wordInt
			wordPos += 1
		corpus.append(sentence)
	return corpus, dict



