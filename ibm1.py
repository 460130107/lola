import numpy as np
import read_data as rd

def ibm1(french, english, initVal, iterations):
	french = rd.open_file(french)
	english = rd.open_file(english)
	corpusFrench, dictFrench = rd.wordToInt(french)
	corpusEnglish, dictEnglish = rd.wordToInt(english)
	tfe = initUniform(initVal, dictFrench, dictEnglish)
	for i in range(0, iterations):
		cfe, ce = eStep(corpusFrench, corpusEnglish, tfe, dictFrench, dictEnglish)
		tfe = mStep(cfe, ce)
	return tfe

# expectation step, calculating counts
def eStep(french, english, tfe, dictFrench, dictEnglish):
	lengthC= len(french)
	cfe = setcfeTo0(dictFrench, dictEnglish)
	ce = setceTo0(dictEnglish)
	for k in range(0, lengthC):
		sEnglish = english[k]
		sFrench = french[k]
		for i in sFrench:
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
def initUniform(initVal, dictFrench, dictEnglish):
	lengthFrench = len(dictFrench)
	lengthEnglish = len(dictEnglish)
	print lengthFrench, lengthEnglish
	initMatrix = np.empty([lengthEnglish, lengthFrench])
	initMatrix.fill(initVal)
	return initMatrix

# Initialize and return a count matrix with 0's
def setcfeTo0(dictFrench, dictEnglish):
	lengthFrench = len(dictFrench)
	lengthEnglish = len(dictEnglish)
	countMatrix = np.zeros((lengthFrench, lengthEnglish))
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
		
#french = 'training\hansards.36.2.f'
#english = 'training\hansards.36.2.e'
french = 'training\example.f'
english = 'training\example.e'

print ibm1(french, english, 0.5, 5)