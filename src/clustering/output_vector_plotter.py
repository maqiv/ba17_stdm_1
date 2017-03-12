from docutils.nodes import header
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import manifold

import numpy as np
import cPickle as pickle
from random import randint

COLOR_VALUES = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000",
                "800000", "008000", "000080", "808000", "800080", "008080", "808080",
                "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0",
                "400000", "004000", "000040", "404000", "400040", "004040", "404040",
                "200000", "002000", "000020", "202000", "200020", "002020", "202020",
                "600000", "006000", "000060", "606000", "600060", "006060", "606060",
                "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
                "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"]


def plot_vectors(X, colors, speaker_names):
    # Next line to silence pyflakes. This import is needed.
    Axes3D

    n_components = 2

    tsne = manifold.TSNE(n_components=n_components, perplexity=50, early_exaggeration=1.0, learning_rate=100,
                         metric="cosine", init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    globalCount = 0
    for i in range(len(set(colors))):
        speaker_entries = colors.count(colors[globalCount])
        plt.scatter(Y[globalCount:globalCount + speaker_entries, 0], Y[globalCount:globalCount + speaker_entries, 1],
                    c=colors[globalCount], label=speaker_names[i], s=40)
        globalCount += speaker_entries
        plt.xlabel('x')
        plt.ylabel('y')
    if len(speaker_names) < 30:
        plt.legend(scatterpoints=1, loc='upper right', ncol=1, fontsize=8)
    plt.show()


def random_colors(n):
    colors = []
    r = lambda: float(randint(0, 1000)) / 1000
    for i in range(n):
        colors.append((r(), r(), r()))
    return colors


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(float(int(value[i:i + lv / 3], 16)) / 255 for i in range(0, lv, lv / 3))


def generate_colors(speakers, length):
    colors = []
    speaker_colors = []
    num_speakers = len(set(speakers))
    if num_speakers <= len(COLOR_VALUES):
        for c in COLOR_VALUES:
            
            speaker_colors.append(hex_to_rgb(c))
    else:
        speaker_colors = random_colors(num_speakers)
    for i in range(length):
        colors.append(speaker_colors[speakers[i]])
    return colors


if __name__ == '__main__':
    with open(
            '/home/patman/pa/1_Code/src/cluster_out_10sp_500ms_256.pickle',
            'rb') as f:
        cluster_output, speakers, speaker_names = pickle.load(f)
        print cluster_output.shape[0]
        colors = generate_colors(speakers, cluster_output.shape[0])

        plot_vectors(cluster_output, colors, speaker_names)

        # print speaker_colors[0:1]
        # plt.imshow([(255, 255, 129)], interpolation='none')
        # plt.legend('speaker 1')
        # plt.show()
