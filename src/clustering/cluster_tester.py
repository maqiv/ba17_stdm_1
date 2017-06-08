import cPickle as pickle
import numpy as np
from sklearn import manifold
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage

'''
Thsi file calculates the MR for a given pair of Cluster Outputs, this code is altered slightly from 
the Bachelor thesis  of Vogt and Lukic (2016).

'''
def misclassification_rate(N, e):
    MR = float(e) / N
    #print float(e)
    return MR


def extract_vectors(num_speakers, vec_size, vectors, y):
    X = np.zeros((num_speakers, vec_size))
    for i in range(num_speakers):
        indices = np.where(y == i)[0]
        outputs = np.take(vectors, indices, axis=0)
        for o in outputs:
            X[i] = np.add(X[i], o)
        X[i] = np.divide(X[i], len(outputs))
    return X, set(y)


def increase_error(indices, e, clusters):
    for i in indices:
        if i < len(e):
            e[i] = 1
        else:
            increase_error(clusters[i], e, clusters)


def calc_MR(X, y, num_speakers, linkage_metric):
    '''
    Generates the Hierarchical Cluster based on the Output of generate_speakr_utterances, with the supplied linkage Metric.
    Based on teh Clustering the MR will be Calculated. Vor the MR calculation an embedding 
    is worng as soon as there is a non matching embedding in the cluster.
    '''
    # cityblock, braycurtis,
    from scipy.spatial.distance import cdist
    X = cdist(X, X, linkage_metric)
    Z = linkage(X, method='complete', metric=linkage_metric)
    clusters = []
    for i in range(len(y)):
        clusters.append([i])
    i = 0
    for z in Z:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

        i += 1

    e = []
    e.append(np.ones(len(y), dtype=np.int))
    print Z.shape
    for z in Z:
        err = list(e[len(e) - 1])
        idx1 = int(z[0])
        idx2 = int(z[1])
       # if idx1 < len(y) and idx2 < len(y) and y[idx1] != y[idx2]:
       #     print y[idx1]
       #     print y[idx2]
        if idx1 >= len(y) or idx2 >= len(y) or y[idx1] != y[idx2]:
            indices = clusters[idx1] + clusters[idx2]
            increase_error(indices, err, clusters)
        else:
            err[idx1] = 0
            err[idx2] = 0
        e.append(err)

    MRs = []
    for err in e:
        MRs.append(misclassification_rate(len(y), sum(err)))

    print 'MR=%f' % np.min(MRs)
    return MRs


def generate_speaker_utterances(train_output, test_output, train_speakers, test_speakers, neuron_number):
    '''
    creates for all Speakers the the Average over their embedings to create an uterance 
    for a each speaker in the Train set (8 sentences), test set (2 sentences)
    the two lists get concatinated to create num_speakers *2 list.
    '''
    num_speakers = len(set(test_speakers))
    print num_speakers
    X_train, y_train = extract_vectors(num_speakers, neuron_number, train_output, train_speakers)
    X_test, y_test = extract_vectors(num_speakers, neuron_number, test_output, test_speakers)
    X = []
    X.extend(X_train)
    X.extend(X_test)
    y = []
    y.extend(y_train)
    y.extend(y_test)

    return X, y, num_speakers


def load_data(train_file, test_file):
    '''
    
    '''
    with open(train_file, 'rb') as f:
        train_output, train_speakers, train_speaker_names = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_output, test_speakers, test_speaker_names = pickle.load(f)
    return train_output, test_output, train_speakers, test_speakers


if __name__ == "__main__":
    PATH = '../../data/experiments/cluster_outputs/lstm_2dense/'
    #TRAIN_FILE = '/media/sf_patrickgerber/cluster_outputs/cluster_output_test_40sp_dense1__2017-06-04_11-52-16.pickle'
    #TEST_FILE = '/media/sf_patrickgerber/cluster_outputs/cluster_output_train_40sp_dense1__2017-06-04_11-52-16.pickle'
    TRAIN_FILE = '/home/patman/PA_Code/data/experiments/cluster_outputs/cluster_output_train_40sp_2017-06-06_22-38-13.pickle'
    TEST_FILE = '/home/patman/PA_Code/data/experiments/cluster_outputs/cluster_output_test_40sp_2017-06-06_22-38-13.pickle'
    train_output, test_output, train_speakers, test_speakers = load_data(TRAIN_FILE, TEST_FILE)


    print train_output.shape
    print test_output.shape
    print set(test_speakers)
    print len(set(train_speakers))
    X, y, num_speakers = generate_speaker_utterances(train_output, test_output, train_speakers, test_speakers, 256)
    print len(X)
    ##print len(y)
    ##print num_speakers
    MRs = calc_MR(X, y, num_speakers, 'cosine')
    plt.plot(MRs, label='40sp', linewidth=2)
#
    #TRAIN_FILE = '/home/sebastian/Dokumente/uni/BT/PA_Code/data/experiments/cluster_outputs/cluster_output_train_60sp_2017-05-09_17-25-19.pickle'
    #TEST_FILE = '/home/sebastian/Dokumente/uni/BT/PA_Code/data/experiments/cluster_outputs/cluster_output_test_60sp_2017-05-09_17-25-19.pickle'
    #train_output, test_output, train_speakers, test_speakers = load_data(TRAIN_FILE, TEST_FILE)
    #X, y, num_speakers = generate_speaker_utterances(train_output, test_output, train_speakers, test_speakers, 128)
    #MRs = calc_MR(X, y, num_speakers, 'cosine')
    #plt.plot(MRs, label='60sp', linewidth=2)
##
    #TRAIN_FILE = '/home/sebastian/Dokumente/uni/BT/PA_Code/data/experiments/cluster_outputs/cluster_output_train_80sp_2017-05-09_17-25-19.pickle'
    #TEST_FILE = '/home/sebastian/Dokumente/uni/BT/PA_Code/data/experiments/cluster_outputs/cluster_output_test_80sp_2017-05-09_17-25-19.pickle'
    #train_output, test_output, train_speakers, test_speakers = load_data(TRAIN_FILE, TEST_FILE)
    #X, y, num_speakers = generate_speaker_utterances(train_output, test_output, train_speakers, test_speakers, 128)
    #MRs = calc_MR(X, y, num_speakers, 'cosine')
    #plt.plot(MRs, label='80sp', linewidth=2)

    #TRAIN_FILE = '/home/patman/pa/1_Code/data/experiments/cluster_outputs/train_cluster_40_kld_lstm2d_layer_lstm_out.pickel'
    #TEST_FILE = '/home/patman/pa/1_Code/data/experiments/cluster_outputs/test_cluster_kld_40_lstm2d_layer_lstm_out.pickel'
    #train_output, test_output, train_speakers, test_speakers = load_data(TRAIN_FILE,TEST_FILE)
    #X, y, num_speakers = generate_speaker_utterances(train_output, test_output, train_speakers, test_speakers, 512)
    #MRs = calc_MR(X, y, num_speakers, 'cosine')
    #plt.plot(MRs, label='lstm2', linewidth=2)

    plt.xlabel('Clusters')
    plt.ylabel('Misclassification Rate (MR)')
    plt.grid()
    plt.legend(loc='lower right', shadow=False)
    plt.ylim(0, 1)
    plt.show()
#
# plt.savefig('/Users/yanicklukic/Google Drive/Carlo+Yanick/BA/experimente/01/diagrams/known_speakers/all_layers_40.png')

# import output_vector_plotter as ovp
#
# tsne = manifold.TSNE(n_components=2, perplexity=30, early_exaggeration=1.0, learning_rate=100, metric="cityblock", init='pca', random_state=10)
# Y_train = tsne.fit_transform(X_train)
# Y_test = tsne.fit_transform(X_test)
#
# for i in range(len(Y_train)):
#     name = train_speaker_names[i]
#     plt.scatter(Y_train[i, 0], Y_train[i, 1], c=ovp.hex_to_rgb(ovp.COLOR_VALUES[i]), label=name, s=50)
#     plt.scatter(Y_test[i, 0], Y_test[i, 1], c=ovp.hex_to_rgb(ovp.COLOR_VALUES[i]), s=50)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(scatterpoints=1, loc='upper right', ncol=1, fontsize=8)
# plt.show()
