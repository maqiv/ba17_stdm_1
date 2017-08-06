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
        labels.append(speaker_names[i])
    return labels


def hierarchical_cluster_plot(cluster_output, speakers, speaker_names):
    X = cdist(cluster_output, cluster_output, 'cosine')
    Z = linkage(X, method='complete', metric='cosine')

    # Assignment of colors to labels
    label_colors = {}
    c = 0
    for name in set(speaker_names):
        label_colors[name] = '#' + ovp.COLOR_VALUES[c]
        c += 1

    num = len(X)
    color = ["#ffccff"] * (2 * num - 1)

    clusters = []
    for speaker in speakers:
        clusters.append([int(speaker)])

    for z in Z:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

    for i in range(len(Z)):
        if int(Z[i, 0]) < len(speakers) and int(Z[i, 1]) < len(speakers) and speakers[int(Z[i, 0])] == speakers[
            int(Z[i, 1])]:
            color[len(speakers) + i] = label_colors[speaker_names[speakers[int(Z[i, 0])]]]
        elif len(set(clusters[len(speakers) + i])) == 1:
            color[len(speakers) + i] = label_colors[speaker_names[clusters[len(speakers) + i][0]]]

    plt.figure(figsize=(25, 10))
    plt.xlabel('speaker name')
    plt.ylabel('distance')
    dendrogram(Z, leaf_rotation=45., leaf_font_size=10., show_contracted=True, link_color_func=lambda x: color[x],
               labels=generate_labels(speakers, speaker_names))

    ax = plt.gca()
    x_labels = ax.get_xmajorticklabels()
    for label in x_labels:
        label.set_color(label_colors[label.get_text()])

    plt.show()


if __name__ == "__main__":
    with open('/home/patman/pa/1_Code/src/test_clustering_10.picklecluster_out_10sp_500ms.pickle', 'rb') as f:
        cluster_output, speakers, speaker_names = pickle.load(f)

    hierarchical_cluster_plot(cluster_output, speakers, speaker_names)
