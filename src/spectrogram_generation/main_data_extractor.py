import cPickle as pickle
import sys

import numpy as np

sys.path.append("/home/patman/pa//1_code")
sys.path.append("/home/patman/pa/1_code/src")
sys.path.append("/home/patman/pa/1_code/src/spectorgram_converter")

import data_extractor as de
from src.core import settings
import src.core.speaker_train_splitter as sts


##### ARGS
MAX_SPEAKERS = 10
WITH_SPLIT = True
SPEAKER_LIST = '../../data/speaker_lists/speakers_10_not_clustering_vs_reynolds.txt'
OUTPUT_1 = '../../data/training/TIMIT_extracted/train_data_10_130ms_2.pickle'
OUTPUT_2 = '../../data/training/TIMIT_extracted/test_data_10_130ms_2.pickle'
###########


ONE_SEC = settings.ONE_SEC # array elements corresponding to one sec
STEP_SIZE = 13
SPECT_DIMENSION = (settings.FREQ_ELEMENTS, ONE_SEC)
MAX_AUDIO_LENGTH = 800

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
print len(np.atleast_1d(y))

if WITH_SPLIT:
    speaker_train_split = sts.SpeakerTrainSplit(0.2, settings.SENTENCES_PER_SPEAKER)
    X_train_valid, X_test, y_train_valid, y_test = speaker_train_split(X, y, None)

    print len(y_train_valid)
    print len(y_test)

    with open(OUTPUT_1, 'wb') as f:
        pickle.dump((X_train_valid, y_train_valid, speaker_names), f, -1)

    with open(OUTPUT_2, 'wb') as f:
        pickle.dump((X_test, y_test, speaker_names), f, -1)
else:
    with open(OUTPUT_1, 'wb') as f:
        pickle.dump((X, y, speaker_names), f, -1)
