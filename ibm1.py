import numpy as np
import read_data as rd

def ibm1(foreign, english, initVal, iterations):
	foreign = rd.open_file(foreign)
	english = rd.open_file(english)
	corpusForeign, dictForeign = rd.wordToInt(foreign)
	corpusEnglish, dictEnglish = rd.wordToInt(english)
	tfe = initUniform(initVal, dictForeign, dictEnglish)
	for i in range(0, iterations):
		cfe, ce = eStep(corpusForeign, corpusEnglish, tfe, dictForeign, dictEnglish)
		tfe = mStep(cfe, ce)
	return tfe

# expectation step, calculating counts
def eStep(foreign, english, tfe, dictForeign, dictEnglish):
	lengthC= len(foreign)
	cfe = setcfeTo0(dictForeign, dictEnglish)
	ce = setceTo0(dictEnglish)
	for k in range(0, lengthC):
		sEnglish = english[k]
		sForeign = foreign[k]
		for i in sForeign:
			for j in sEnglish:
				tfeTotal = sumWord(i, sEnglish, tfe)
				delta = tfe[j,i]/tfeTotal
				cfe[j,i] += delta
				ce[j,0] += delta
	return cfe, ce

# Normalizing counts
def mStep(cfe, ce):
	return cfe/ce

# Initialize and return a t(f|e) parameter with uniform initializations		
def initUniform(initVal, dictForeign, dictEnglish):
	lengthForeign = len(dictForeign)
	lengthEnglish = len(dictEnglish)
	initMatrix = np.empty([lengthEnglish, lengthForeign])
	initMatrix.fill(initVal)
	return initMatrix

# Initialize and return a count matrix with 0's
def setcfeTo0(dictForeign, dictEnglish):
	lengthForeign = len(dictForeign)
	lengthEnglish = len(dictEnglish)
	countMatrix = np.zeros((lengthForeign, lengthEnglish))
	return countMatrix
	
def setceTo0(dict):
	length = len(dict)
	countMatrix = np.zeros((length, 1))
	return countMatrix

def sumWord(i, sEnglish,  tfe):
	total = 0
	for  j in sEnglish:
		total += tfe[j, i]
	return total
		
#french = rd.open_file('training\hansards.36.2.f')
#english = rd.open_file('training\hansards.36.2.e')
french = 'training\example.f'
english = 'training\example.e'

print ibm1(french, english, 0.5, 10000)