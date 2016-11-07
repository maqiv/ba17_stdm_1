from random import randint

import numpy as np

from core import settings


def extract_spectrogram(spectrogram):
    zeros = 0
    for x in spectrogram[0]:
        if x == 0.0:
            zeros += 1
        else:
            zeros = 0
    while spectrogram.shape[1] - zeros < settings.ONE_SEC:
        zeros -= 1
    spect = spectrogram[0:settings.FREQ_ELEMENTS, 0:spectrogram.shape[1] - zeros]
    return spect

def transform(X, y) :
	s = X.shape[0] / settings.NUM_OF_SPEAKERS
	p = X.shape[0] * X.shape[3]
	Xb = np.zeros((p, 1, settings.FREQ_ELEMENTS, settings.ONE_SEC), dtype=np.float32)
	yb = np.zeros(p, dtype=np.int32)
	for ()
	sp = extract_spectrogram(X[1, 0])
	

	#for j in (0 : p) 
	#spect = extract_spectrogram([j, 0])
	#print 
	#Xb[j] = 






