# valid_speakers = []
# with open('data/speakers100_50w_50m.txt', 'rb') as f:
#     for line in f:
#         valid_speakers.append(line.replace('\n', ''))
#
# counter = 0
# oldSpeaker = ''
# for root, directories, filenames in os.walk('/Users/yanicklukic/Repositories/PA/projektarbeit_code/data/TIMIT/Train'):
#     for filename in filenames:
#         if '_RIFF.WAV' in filename and oldSpeaker != root[-5:] and root[-5:-4] == 'M' and counter < 50:
#             oldSpeaker = root[-5:]
#             print root[-5:]
#             counter += 1


import cPickle as pickle
import sys

sys.path.append("/home/patman/pa/BA/1_Code")
sys.path.append("/home/patman/pa/BA/1_Code/src")
sys.path.append("/home/patman/pa/BA/1_Code/src/spectorgram_converter")
from matplotlib import pyplot as plt

from src.core import settings
import spectrogram_converter as sc


path = '../../data/training/TIMIT/TRAIN/DR2/MKJO0/SA1_RIFF.WAV'
spect = sc.mel_spectrogram(path)
print sc.duration(path)
print spect.shape

from scipy.io.wavfile import read
samprate, wavdata = read(path)
import numpy as np
chunks = np.array_split(wavdata, 155)
dbs = [20*np.log10(np.sqrt(np.mean(chunk**2))) for chunk in chunks]
print dbs

#plt.figure(2)
plt.imshow(spect[:, 20:160])
#plt.show()

#specshow(spect)
cbar = plt.colorbar()
n = np.linspace(0, 35, num=11)
labels=[]
for l in n:
    labels.append(str(l) + ' dB')
cbar.ax.set_yticklabels(labels)

plt.xlabel('Spektra (in Zeit)')
plt.ylabel('Frequenz-Datenpunkte')

#plt.imshow(spect)

plt.show()