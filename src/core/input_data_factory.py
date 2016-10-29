import numpy as np

from src.core import settings


def generate_cluster_data(X, y, overlapping=False):
    X_cluster = np.zeros((10000, 1, settings.FREQ_ELEMENTS, settings.ONE_SEC), dtype=np.float32)
    y_cluster = []

    step = settings.ONE_SEC
    if overlapping:
        step = settings.ONE_SEC / 2
    pos = 0
    for i in range(len(X)):
        spect = extract_spectrogram(X[i, 0])

        for j in range(spect.shape[1] / step):
            y_cluster.append(y[i])
            seg_idx = j * step
            try:
                X_cluster[pos, 0] = spect[:, seg_idx:seg_idx + settings.ONE_SEC]
            except ValueError:
                # if the last segment doesn't match ignore it
                pass
            pos += 1

    return X_cluster[0:len(y_cluster)], np.asarray(y_cluster, dtype=np.int32)


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
