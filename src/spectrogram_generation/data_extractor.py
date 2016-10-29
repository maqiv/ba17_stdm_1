import numpy as np
import scipy.io.wavfile as wav
import os
from random import shuffle
import spectrogram_converter as sc


class DataExtractor:
    def __init__(self, max_speakers, one_sec, step_size, spect_dimension):
        self.max_speakers = max_speakers
        self.one_sec = one_sec
        self.step_size = step_size
        self.spect_dimension = spect_dimension

    def extract_mel_spectrogram(self, wav_path, X, y, next_idx, curr_speaker_num):
        Sxx = sc.mel_spectrogram(wav_path)
        for i in range(Sxx.shape[0]):
            for j in range(Sxx.shape[1]):
                X[next_idx, 0, i, j] = Sxx[i, j]
        y[next_idx] = curr_speaker_num
        return 1

    def extract_mel_spectrogram_segmented(self, wav_path, X, y, next_idx, curr_speaker_num):
        mat = sc.spectrogram(wav_path)
        return self.extract(mat, X, y, next_idx, curr_speaker_num)

    def extract(self, mat, X, y, next_idx, curr_speaker_num):
        i = 0
        width = mat.shape[1]
        while width - i * self.step_size > self.one_sec:
            c = i * self.step_size
            X[next_idx + i, 0] = mat[:, c:c + self.one_sec]
            y[next_idx + i] = curr_speaker_num
            i += 1
        # add last snippet
        pos = i * self.one_sec
        if pos != width:
            step_back = self.one_sec - (width - pos)
            X[next_idx + i, 0] = mat[:, pos - step_back:mat.shape[1]]
            y[next_idx + i] = curr_speaker_num

        return i

    def traverse_TIMIT_data(self, base_folder, X, y, valid_speakers):
        speaker_names = []

        global_idx = 0
        curr_speaker_num = -1
        speaker = curr_speaker_num
        old_speaker = ''
        for root, directories, filenames in os.walk(base_folder):
            for filename in filenames:
                # extract speaker
                if '_RIFF.WAV' in filename and root[-5:] in valid_speakers:
                    speaker = root[-5:]
                    if speaker != old_speaker:
                        curr_speaker_num += 1
                        old_speaker = speaker
                        speaker_names.append(speaker)
                        print 'Extraction progress: %d/%d' % (curr_speaker_num+1, self.max_speakers)

                    if curr_speaker_num < self.max_speakers:
                        full_path = os.path.join(root, filename)
                        global_idx += self.extract_mel_spectrogram(full_path, X, y, global_idx, curr_speaker_num)
        return X[0:global_idx], y[0:global_idx], speaker_names

    def list_shuffle(self, X, y):
        X_shuf = np.zeros(X.shape, dtype=np.float32)
        y_shuf = np.zeros(y.shape, dtype=np.int32)
        index_shuf = range(len(y_shuf))
        shuffle(index_shuf)
        j = 0
        for i in index_shuf:
            X_shuf[j, 0] = X[i, 0]
            y_shuf[j] = y[i]
            j += 1
        return X_shuf, y_shuf
