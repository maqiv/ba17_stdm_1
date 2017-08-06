import wave

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import librosa


def spectrogram(wav_file):
    (rate, sig) = wav.read(wav_file)

    nperseg = 20*rate/1000;
    for i in range(0, 12):
        n = 2**i
        if n >= nperseg:
            nfft = n
            break

    f, t, Sxx = signal.spectrogram(sig, fs=rate, window='hamming', nperseg=nperseg, noverlap=nperseg/2,
                nfft=nfft, detrend=None, scaling='spectrum', return_onesided=True)

    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            Sxx[i, j] = hr_to_mel_spect(Sxx[i, j])

    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            Sxx[i, j] = dyn_range_compression(Sxx[i, j])

    return Sxx


def duration(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.getnframes()
    rate = wav.getframerate()
    dur = frames / float(rate)
    #print 'duration: %f' % duration
    wav.close()
    return dur

def dyn_range_compression(x):
    return np.log10(1 + 10000 * x)

def hr_to_mel_spect(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log10(f/700 +1)

def mel_spectrogram(wav_file):
    y, sr = librosa.load(path=wav_file, sr=None)
    nperseg = 10*sr/1000;
    Sxx = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=nperseg)

    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            Sxx[i, j] = dyn_range_compression(Sxx[i, j])

    return Sxx


# import matplotlib.pyplot as plt
#
# melspect = mel_spectrogram('data/SA1_RIFF.WAV')
# print melspect.shape
# spect = spectrogram('data/SA1_RIFF.WAV')
# print spect.shape
#
# f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
# ax1.imshow(melspect)
# ax2.imshow(spect)
# plt.show()
