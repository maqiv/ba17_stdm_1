import matplotlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import numpy as np
import cPickle as pickle
import output_vector_plotter as ovp


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

    TRAIN_FILE = 'train_cluster_out_40sp_500.pickle'
    TEST_FILE = 'test_cluster_out_40sp_500.pickle'
with open('/home/patman/pa/1_Code/src/train_cluster_out_40sp__256_500.pickle', 'rb') as f:
    test_output, test_speakers, speaker_names = pickle.load(f)

with open('/home/patman/pa/1_Code/src/test_cluster_out_40sp__256_500.pickle', 'rb') as f:
    train_output, train_speakers, _ = pickle.load(f)

NUM_SPEAKERS = len(speaker_names)

X_train, y_train = extract_vectors(NUM_SPEAKERS, 512, train_output, train_speakers)
X_test, y_test = extract_vectors(NUM_SPEAKERS, 512, test_output, test_speakers)
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
plt.xlabel('speaker name', fontsize=18)
plt.ylabel('distance', fontsize=18)
plt.tick_params(labelsize=18)
dendrogram(Z, leaf_rotation=45., leaf_font_size=18., show_contracted=True, link_color_func=lambda x: color[x], labels=generate_labels(speakers, speaker_names))

ax = plt.gca()
x_labels = ax.get_xmajorticklabels()
for label in x_labels:
    label.set_color(label_colors[label.get_text()])

plt.show()
print speaker_names[36]
print speaker_names[0]
print speaker_names[10]
print speaker_names[29]