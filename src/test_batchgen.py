import os
import csv
import json
import time
import logging
import datetime
import numpy as np
import tensorflow as tf

import core.data_gen as dg
import core.pairwise_kl_divergence_np as pkld
import cPickle as pickle

with open('/home/patman/pa/1_Code/data/training/TIMIT_extracted/speakers_100_50w_50m_not_reynolds.pickle', 'rb') as f:
	(X, y, speaker_names) = pickle.load(f)
X_t, X_v, y_t, y_v = dg.splitter(X, y, 0.125, 8)

def build_pairs(l):
	same = 0.
	pairs = 0.
	print l
	for i in range(len(l)):
		for j in range(len(l)):
			if (i < j):
				pairs = pairs +1
				if (l[i] == l[j]):
					print l[i]
					print l[j]
					same = same+ 1
	print "same pairs: ",same
	print "pairs : ", pairs
	return same/pairs				





train_gen = dg.batch_generator_lstm_v2(X_t, y_t, batch_size=100, segment_size=50)

s = 0
for i in range(1):
	x , y = train_gen.next()
	p = build_pairs(y)
	print p
	s += p

print p/100