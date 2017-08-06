import os
import csv
import json
import time
import logging
import datetime
import numpy as np

import core.data_gen as dg
import core.pairwise_kl_divergence_np as pkld
import cPickle as pickle

''' 
checks the generated batch for the amount of pairs from the same speaker.

'''

with open('/home/patman/pa/1_Code/data/training/TIMIT_extracted/speakers_100_50w_50m_not_reynolds.pickle', 'rb') as f:
	(X, y, speaker_names) = pickle.load(f)
X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)

def build_pairs(l):
	same = 0.
	pairs = 0.
	pair_list = []
	print l
	for i in range(len(l)):
		j = i+1
		for j in range(i+1,len(l)):
				print j
				pairs = pairs +1
				if (l[i] == l[j]):
					same = same+ 1
					pair_list.append((l[i], l[j], i , j, 1))
				else:
					pair_list.append((l[i], l[j], i , j, 0))
	print pair_list
	print np.vstack(pair_list).shape
	return same/pairs				





train_gen = dg.batch_generator_lstm_v2(X_t, y_t, batch_size=100, segment_size=50)

s = 0
for i in range(1):
	x , y = train_gen.next()
	p = build_pairs(y)
	print p
	s += p

print s/100
