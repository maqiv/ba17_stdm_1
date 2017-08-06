import matplotlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import numpy as np
import cPickle as pickle
import output_vector_plotter as ovp
import sys


def generate_labels(speakers, speaker_names):
    labels = []
    old_speaker = -1
    i = -1
    for speaker in speakers:
        if speaker != old_speaker:
            old_speaker = speaker
            i += 1
        if i == len(speaker_names):
            i = 0
        labels.append(speaker_names[i])
    return labels


def extract_vectors(num_speakers, vec_size, vectors, y):
    X = np.zeros((num_speakers, vec_size))
    for i in range(num_speakers):
        indices = np.where(y == i)[0]
        outputs = np.take(vectors, indices, axis=0)
        for o in outputs:
            X[i] = np.add(X[i], o)
        X[i] = np.divide(X[i], len(outputs))

    return X, set(y)


#with open('data/cluster_5_output_06112015001.pickle', 'rb') as f:
#    cluster_output, speakers, speaker_names = pickle.load(f)

TRAIN_FILE = '/home/sebastian/Dokumente/uni/BT/PA_Code/data/experiments/cluster_outputs/cluster_output_train_40sp_' + sys.argv[2] + '.pickle'
TEST_FILE = '/home/sebastian/Dokumente/uni/BT/PA_Code/data/experiments/cluster_outputs/cluster_output_test_40sp_' + sys.argv[2] + '.pickle'
with open(TEST_FILE, 'rb') as f:
    test_output, test_speakers, speaker_names = pickle.load(f)

with open(TRAIN_FILE, 'rb') as f:
    train_output, train_speakers, _ = pickle.load(f)

NUM_SPEAKERS = len(speaker_names)

X_train, y_train = extract_vectors(NUM_SPEAKERS, int(sys.argv[1]), train_output, train_speakers)
X_test, y_test = extract_vectors(NUM_SPEAKERS, int(sys.argv[1]), test_output, test_speakers)
X = []
X.extend(X_train)
X.extend(X_test)
speakers = []
speakers.extend(y_train)
speakers.extend(y_test)


X = cdist(X, X, 'cosine')
Z = linkage(X, method='complete', metric='cosine')
print Z.shape
# Assignment of colors to labels
label_colors = {}
c=0
for name in speaker_names:
    label_colors[name] = '#'+ovp.COLOR_VALUES[c]
    c += 1

num = len(X)
color = ["#ffccff"] * (2 * num - 1)

clusters = []
for speaker in speakers:
    clusters.append([int(speaker)])

    print len(clusters)
i = 0
for z in Z:
    clusters.append(clusters[int(z[0])] + clusters[int(z[1])])
    print clusters[80+i]
    i += 1

for i in range(len(Z)):
    if int(Z[i, 0]) < len(speakers) and int(Z[i, 1]) < len(speakers) and speakers[int(Z[i, 0])] == speakers[int(Z[i, 1])]:
        color[len(speakers)+i] = label_colors[speaker_names[speakers[int(Z[i, 0])]]]
    elif len(set(clusters[len(speakers)+i])) == 1:
        color[len(speakers)+i] = label_colors[speaker_names[clusters[len(speakers)+i][0]]]

matplotlib.rcParams['lines.linewidth'] = 4
plt.figure(figsize=(25, 10))
plt.xlabel('Sprecher', fontsize=18)
plt.ylabel('Distanz', fontsize=18)
plt.axhline(y=0.01055, c='k')
plt.tick_params(labelsize=18)
dendrogram(Z, leaf_rotation=90., leaf_font_size=10., show_contracted=True, link_color_func=lambda x: color[x], labels=generate_labels(speakers, speaker_names))

ax = plt.gca()
x_labels = ax.get_xmajorticklabels()
for label in x_labels:
    label.set_color(label_colors[label.get_text()])

plt.show()
print speaker_names[36]
print speaker_names[0]
print speaker_names[10]
print speaker_names[29]
