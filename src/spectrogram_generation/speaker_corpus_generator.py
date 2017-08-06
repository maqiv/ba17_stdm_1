import os
from random import randint

from src.core import settings

with_not_to_use = True
num_speakers_clustering = 40
OUTPUT_FILE = '../../data/speaker_lists/speakers_40_not_clustering_vs_reynolds.txt'
FORBIDDEN_SPEAKERS = '../../data/speaker_lists/speakers_40_clustering_vs_reynolds.txt'

speakers = []
oldSpeaker = ''
for root, directories, filenames in os.walk('../../data/training/TIMIT/'):
    for filename in filenames:
        if '_RIFF.WAV' in filename and oldSpeaker != root[-5:]:
            oldSpeaker = root[-5:]
            speakers.append(root[-5:])

y = []
if with_not_to_use:
    print 'Checking for double users.'
    not_to_use = []
    with open(FORBIDDEN_SPEAKERS, 'rb') as f:
        not_to_use = f.read().splitlines()

    while len(y) < num_speakers_clustering:
        idx = randint(0, len(speakers)-1)
        speaker = speakers.pop(idx)
        if speaker not in not_to_use:
            y.append(speaker)
else:
    print 'Ignoring double users.'
    for i in range(settings.NUM_OF_SPEAKERS):
        idx = randint(0, len(speakers))
        y.append(speakers.pop(idx))

wCount = 0
mCount = 0
with open(OUTPUT_FILE, 'wb') as f:
    for speaker in y:
        f.write(speaker)
        f.write('\n')
        if speaker[-5:-4] == 'M':
            mCount += 1
        else:
            wCount += 1

print 'Successfully generated. %d women, %d men' % (wCount, mCount)