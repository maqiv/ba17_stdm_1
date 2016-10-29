import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import wave
import scipy.io.wavfile as wav
import spectrogram_converter
from features import logfbank
import cPickle as pickle


def duration(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.getnframes()
    rate = wav.getframerate()
    dur = frames / float(rate)
    #print 'duration: %f' % duration
    wav.close()
    return dur

def func(x):
    return np.log(1 + 10000 * x)

'''
wav_file = 'data/SI1400_RIFF.WAV'
(rate, sig) = wav.read(wav_file)
duration(wav_file)

f, t, Sxx = signal.spectrogram(sig, window=('gaussian', 128),scaling='spectrum', return_onesided=True)

for i in range(Sxx.shape[0]):
    for j in range(Sxx.shape[1]):
        Sxx[i, j] = func(Sxx[i, j])

print Sxx.shape
'''

with open('data/spectrogram_gabriel.pickle', 'rb') as f:
    (X_gab, y_gab) = pickle.load(f)

wav_file = 'data/SA1_RIFF.WAV'

spect_new = spectrogram_converter.spectrogram(wav_file)
(rate, sig) = wav.read(wav_file)
fbe = logfbank(sig, samplerate=rate, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
fbe = np.fliplr(zip(*fbe[::-1]))

print 'Duration: %f' % duration(wav_file)
print 'Array Size Spect: %d' % spect_new.shape[2]
print 'Array Size FBE new: %d' % fbe.shape[1]
print 'Array Size FBE old: %d' % X_gab.shape[3]

f, (plt1, plt2, plt3) = plt.subplots(3, 1, sharey=False)
plt1.imshow(spect_new)
plt1.set_title('new spectrogram')

plt2.imshow(fbe)
plt2.set_title('new FBE')

plt3.imshow(np.hstack([X_gab[0, 0], X_gab[1, 0], X_gab[2, 0], X_gab[3, 0], X_gab[4, 0]]))
plt3.set_title('old FBE')

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()