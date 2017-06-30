import cPickle as pickle
import sys

import numpy as np
import data_extractor as de
import speaker_train_splitter as sts


##### ARGS
SENTENCES_PER_SPEAKER = 10
FREQ_ELEMENTS = 128
MAX_SPEAKERS = 5
WITH_SPLIT = True
SPEAKER_LIST = '../../data/speaker_lists/speakers_5_clustering_vs_reynolds_v3.txt'
#OUTPUT_1 = '../../data/training/TIMIT_extracted/train_speakers_5_clustering_vs_reynolds_v3.pickle'
#OUTPUT_2 = '../../data/training/TIMIT_extracted/test_speakers_5_clustering_vs_reynolds_v3.pickle'
OUTPUT_1 = '../../data/training/TIMIT_extracted/train_speakers_5_clustering_vs_reynolds.pickle'
OUTPUT_2 = '../../data/training/TIMIT_extracted/test_speakers_5_clustering_vs_reynolds.pickle'
###########


ONE_SEC = 100 # array elements corresponding to one sec
STEP_SIZE = 100
SPECT_DIMENSION = (FREQ_ELEMENTS, ONE_SEC)
MAX_AUDIO_LENGTH = 100

extractor = de.DataExtractor(MAX_SPEAKERS, ONE_SEC, STEP_SIZE, SPECT_DIMENSION)
#X = np.zeros((MAX_SPEAKERS*1000, 1, SPECT_DIMENSION[0], SPECT_DIMENSION[1]), dtype=np.float32)
X = np.zeros((MAX_SPEAKERS*20, 1, SPECT_DIMENSION[0], MAX_AUDIO_LENGTH), dtype=np.float32)
y = np.zeros(MAX_SPEAKERS*20, dtype=np.int32)

valid_speakers = []
with open(SPEAKER_LIST, 'rb') as f:
    for line in f:
        valid_speakers.append(line.replace('\n', ''))

X, y, speaker_names = extractor.traverse_TIMIT_data('../../data/training/TIMIT/', X, y, valid_speakers)

print X.shape

if WITH_SPLIT:
    speaker_train_split = sts.SpeakerTrainSplit(0.2, SENTENCES_PER_SPEAKER)
    X_train_valid, X_test, y_train_valid, y_test = speaker_train_split(X, y, None)

    print len(y_train_valid)
    print len(y_test)
    print y_test
    print y_train_valid

    with open(OUTPUT_1, 'wb') as f:
        pickle.dump((X_train_valid, y_train_valid, speaker_names), f, -1)

    with open(OUTPUT_2, 'wb') as f:
        pickle.dump((X_test, y_test, speaker_names), f, -1)
else:
    with open(OUTPUT_1, 'wb') as f:
        pickle.dump((X, y, speaker_names), f, -1)
